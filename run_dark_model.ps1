Set-Location 'C:\Users\radar\SolarPipe\.claude\worktrees\eloquent-benz'
$py = 'C:\Users\radar\AppData\Local\Programs\Python\Python312\python.exe'
$log = 'C:\Users\radar\SolarPipe\logs\dark_model.log'
New-Item -Force -ItemType File $log | Out-Null
& $py -u scripts/train_dark_model.py --write-oof-preds 2>&1 | Tee-Object -FilePath $log
$ec = $LASTEXITCODE
Add-Content $log "EXIT_CODE: $ec"
Write-Host "EXIT_CODE: $ec"
