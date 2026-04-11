$py = 'C:\Users\radar\AppData\Local\Programs\Python\Python312\python.exe'
$wt = 'C:\Users\radar\SolarPipe\.claude\worktrees\eloquent-benz'
& $py -m py_compile "$wt\scripts\train_dark_model.py"; Write-Host "dark_model: exit $LASTEXITCODE"
& $py -m py_compile "$wt\scripts\sweep_tft_hyperparams.py"; Write-Host "sweep: exit $LASTEXITCODE"
& $py -m py_compile "$wt\scripts\build_nnls_ensemble.py"; Write-Host "ensemble: exit $LASTEXITCODE"
