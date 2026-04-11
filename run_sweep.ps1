Set-Location 'C:\Users\radar\SolarPipe\.claude\worktrees\eloquent-benz'
$py = 'C:\Users\radar\AppData\Local\Programs\Python\Python312\python.exe'
$log = 'C:\Users\radar\SolarPipe\logs\sweep_tft.log'
New-Item -Force -ItemType File $log | Out-Null
& $py -u scripts/sweep_tft_hyperparams.py --n-configs 20 --device cuda --seed 42 2>&1 | Tee-Object -FilePath $log
$ec = $LASTEXITCODE
Add-Content $log "EXIT_CODE: $ec"
Write-Host "EXIT_CODE: $ec"
