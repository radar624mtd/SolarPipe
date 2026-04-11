Set-Location 'C:\Users\radar\SolarPipe\.claude\worktrees\eloquent-benz'
$py = 'C:\Users\radar\AppData\Local\Programs\Python\Python312\python.exe'
$log = 'C:\Users\radar\SolarPipe\logs\tft_expanded.log'
New-Item -Force -ItemType File $log | Out-Null
# -u = unbuffered stdout/stderr so Tee-Object gets lines as they print
& $py -u scripts/train_tft_model.py --device cuda --epochs 150 2>&1 | Tee-Object -FilePath $log
$ec = $LASTEXITCODE
Add-Content $log "EXIT_CODE: $ec"
Write-Host "EXIT_CODE: $ec"
