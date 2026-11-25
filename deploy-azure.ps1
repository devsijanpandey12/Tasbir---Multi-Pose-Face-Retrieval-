<#
.SYNOPSIS
    Deploys Tasbir (FastAPI ONNX backend + Next.js frontend) to Azure Container Apps.
    Uses local Docker build + ACR push (compatible with Azure for Students) and
    a WorkloadProfiles Consumption environment with Azure Files persistent storage.
#>

param(
    [string]$ResourceGroup = "rg-tasbir",
    [string]$Location = "eastasia",
    [string]$AcrName = "acrtasbirsijan",
    [string]$StorageAccount = "sttasbirsijan",
    [string]$EnvName = "cae-tasbir-wp"
)

$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "==> 1/7 Ensuring Azure CLI extension and resource group ($ResourceGroup in $Location)..." -ForegroundColor Cyan
az extension add --name containerapp --upgrade -y | Out-Null
az group create --name $ResourceGroup --location $Location --output none

Write-Host "==> 2/7 Ensuring Azure Container Registry ($AcrName) and Docker login..." -ForegroundColor Cyan
$acrExists = az acr show --name $AcrName --resource-group $ResourceGroup --query name -o tsv 2>$null
if (-not $acrExists) {
    az acr create `
        --resource-group $ResourceGroup `
        --name $AcrName `
        --sku Basic `
        --admin-enabled true `
        --location $Location `
        --output none
}

$acrServer = az acr show --name $AcrName --query loginServer -o tsv
$acrUser = az acr credential show --name $AcrName --query username -o tsv
$acrPass = az acr credential show --name $AcrName --query "passwords[0].value" -o tsv
$acrPass | docker login $acrServer --username $acrUser --password-stdin | Out-Null

Write-Host "==> 3/7 Ensuring Storage Account ($StorageAccount) + Azure Files share (tasbir-data)..." -ForegroundColor Cyan
$stgExists = az storage account show --name $StorageAccount --resource-group $ResourceGroup --query name -o tsv 2>$null
if (-not $stgExists) {
    az storage account create `
        --resource-group $ResourceGroup `
        --name $StorageAccount `
        --location $Location `
        --sku Standard_LRS `
        --kind StorageV2 `
        --output none
}

$storageKey = az storage account keys list `
    --resource-group $ResourceGroup `
    --account-name $StorageAccount `
    --query "[0].value" -o tsv

az storage share-rm create `
    --resource-group $ResourceGroup `
    --storage-account $StorageAccount `
    --name "tasbir-data" `
    --quota 5 `
    --output none

Write-Host "==> 4/7 Ensuring Container Apps Environment ($EnvName) and binding persistent storage..." -ForegroundColor Cyan
$envExists = az containerapp env show --name $EnvName --resource-group $ResourceGroup --query name -o tsv 2>$null
if (-not $envExists) {
    az containerapp env create `
        --name $EnvName `
        --resource-group $ResourceGroup `
        --location $Location `
        --environment-mode WorkloadProfiles `
        --output none
}

az containerapp env storage set `
    --name $EnvName `
    --resource-group $ResourceGroup `
    --storage-name "tasbirdata" `
    --azure-file-account-name $StorageAccount `
    --azure-file-account-key $storageKey `
    --azure-file-share-name "tasbir-data" `
    --access-mode ReadWrite `
    --output none

Write-Host "==> 5/7 Building and pushing FastAPI ONNX backend (tasbir-api)..." -ForegroundColor Cyan
$tag = (Get-Date -Format "yyyyMMddHHmmss")
docker build --platform linux/amd64 -t "$acrServer/tasbir-api:$tag" -t "$acrServer/tasbir-api:latest" "$Root\backend"
docker push "$acrServer/tasbir-api:$tag"
docker push "$acrServer/tasbir-api:latest"

$apiExists = az containerapp show --name "tasbir-api" --resource-group $ResourceGroup --query name -o tsv 2>$null
if (-not $apiExists) {
    az containerapp create `
        --name "tasbir-api" `
        --resource-group $ResourceGroup `
        --environment $EnvName `
        --image "$acrServer/tasbir-api:$tag" `
        --registry-server $acrServer `
        --registry-username $acrUser `
        --registry-password $acrPass `
        --target-port 8000 `
        --ingress external `
        --cpu 2.0 `
        --memory 4.0Gi `
        --min-replicas 0 `
        --max-replicas 1 `
        --env-vars "TASBIR_DATA_DIR=/app/data" "TASBIR_MODELS_DIR=/app/models/buffalo_l" "TASBIR_SQLITE_NOLOCK=true" `
        --output none
} else {
    az containerapp update `
        --name "tasbir-api" `
        --resource-group $ResourceGroup `
        --image "$acrServer/tasbir-api:$tag" `
        --output none
}

# Mount Azure Files volume at /app/data via YAML patch
$apiYaml = "$env:TEMP\tasbir-api-export.yaml"
az containerapp show --name "tasbir-api" --resource-group $ResourceGroup --output yaml | Set-Content -Path $apiYaml
& "$Root\backend\.venv\Scripts\python.exe" -c @"
import yaml, pathlib
p = pathlib.Path(r'$apiYaml')
doc = yaml.safe_load(p.read_text(encoding='utf-8'))
tpl = doc['properties']['template']
vols = tpl.get('volumes') or []
if not any(v.get('name') == 'tasbir-data-vol' for v in vols):
    vols.append({'name': 'tasbir-data-vol', 'storageName': 'tasbirdata', 'storageType': 'AzureFile'})
tpl['volumes'] = vols
c0 = tpl['containers'][0]
mounts = c0.get('volumeMounts') or []
if not any(m.get('volumeName') == 'tasbir-data-vol' for m in mounts):
    mounts.append({'volumeName': 'tasbir-data-vol', 'mountPath': '/app/data'})
c0['volumeMounts'] = mounts
p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')
"@
az containerapp update --name "tasbir-api" --resource-group $ResourceGroup --yaml $apiYaml --output none

$apiFqdn = az containerapp show --name "tasbir-api" --resource-group $ResourceGroup --query "properties.configuration.ingress.fqdn" -o tsv
$apiUrl = "https://$apiFqdn"
Write-Host "    Backend live at: $apiUrl" -ForegroundColor Green

Write-Host "==> 6/7 Building and pushing Next.js frontend (tasbir-web)..." -ForegroundColor Cyan
docker build --platform linux/amd64 `
    --build-arg "NEXT_PUBLIC_API_URL=$apiUrl" `
    -t "$acrServer/tasbir-web:$tag" `
    -t "$acrServer/tasbir-web:latest" `
    "$Root\frontend"
docker push "$acrServer/tasbir-web:$tag"
docker push "$acrServer/tasbir-web:latest"

$webExists = az containerapp show --name "tasbir-web" --resource-group $ResourceGroup --query name -o tsv 2>$null
if (-not $webExists) {
    az containerapp create `
        --name "tasbir-web" `
        --resource-group $ResourceGroup `
        --environment $EnvName `
        --image "$acrServer/tasbir-web:$tag" `
        --registry-server $acrServer `
        --registry-username $acrUser `
        --registry-password $acrPass `
        --target-port 3000 `
        --ingress external `
        --cpu 0.5 `
        --memory 1.0Gi `
        --min-replicas 0 `
        --max-replicas 1 `
        --output none
} else {
    az containerapp update `
        --name "tasbir-web" `
        --resource-group $ResourceGroup `
        --image "$acrServer/tasbir-web:$tag" `
        --output none
}

$webFqdn = az containerapp show --name "tasbir-web" --resource-group $ResourceGroup --query "properties.configuration.ingress.fqdn" -o tsv
$webUrl = "https://$webFqdn"

Write-Host "==> 7/7 Wiring backend CORS to allow $webUrl..." -ForegroundColor Cyan
az containerapp update `
    --name "tasbir-api" `
    --resource-group $ResourceGroup `
    --set-env-vars "TASBIR_CORS_ORIGINS=$webUrl,http://localhost:3000" `
    --output none

Write-Host ""
Write-Host "Deployment complete!" -ForegroundColor Yellow
Write-Host "  Web App : $webUrl" -ForegroundColor Green
Write-Host "  API Docs: $apiUrl/docs" -ForegroundColor Cyan
