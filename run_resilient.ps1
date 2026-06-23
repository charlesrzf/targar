# Wrapper resiliente: re-executa a Fase 2->6 (resumível) até concluir.
# Saída CRUA para log (sem Select-String, sem buffering).
$py  = "C:\Users\user\anaconda3\envs\cu-targeting\python.exe"
$log = "D:\argentina\logs\pipeline_run.log"
Set-Location "D:\argentina\pipeline"
for ($i = 1; $i -le 30; $i++) {
    "=== TENTATIVA $i @ $(Get-Date -Format 'HH:mm:ss') ===" | Out-File -Append -Encoding utf8 $log
    & $py run_pipeline.py --start-from 2 --force *>> $log
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        "=== PIPELINE OK @ $(Get-Date -Format 'HH:mm:ss') ===" | Out-File -Append -Encoding utf8 $log
        break
    }
    "=== exit=$code @ $(Get-Date -Format 'HH:mm:ss') — resume em 20s ===" | Out-File -Append -Encoding utf8 $log
    Start-Sleep 20
}
