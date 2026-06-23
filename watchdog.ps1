# Watchdog: mata python se o log ficar parado > limite (hang em leitura S3).
# O wrapper resiliente então reinicia e retoma. Encerra quando pipeline conclui.
$log    = "D:\argentina\logs\pipeline_run.log"
$thresh = 1500   # 25 min sem nova linha = hang
while ($true) {
    Start-Sleep 300
    if (-not (Test-Path $log)) { continue }
    $tail = Get-Content $log -Tail 8 -ErrorAction SilentlyContinue
    if ($tail -match "PIPELINE OK") {
        "WATCHDOG @ $(Get-Date -Format 'HH:mm:ss'): pipeline concluido — encerrando watchdog" | Out-File -Append -Encoding utf8 $log
        break
    }
    $age = ((Get-Date) - (Get-Item $log).LastWriteTime).TotalSeconds
    if ($age -gt $thresh) {
        $py = Get-Process python -ErrorAction SilentlyContinue
        if ($py) {
            "WATCHDOG @ $(Get-Date -Format 'HH:mm:ss'): log parado $([int]$age)s (>$thresh) — matando python (hang)" | Out-File -Append -Encoding utf8 $log
            $py | Stop-Process -Force
        } else {
            "WATCHDOG @ $(Get-Date -Format 'HH:mm:ss'): idle e sem python — encerrando watchdog" | Out-File -Append -Encoding utf8 $log
            break
        }
    }
}
