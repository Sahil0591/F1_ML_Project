param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [pscredential]$Credential,
    [switch]$Preview
)

$ErrorActionPreference = 'Stop'
$resolvedRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$pythonPath = Join-Path $resolvedRoot '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw 'Windowless Python is missing from the virtual environment.' }
$taskName = 'f1_ml_predictor_prospective'
$arguments = '-m f1_ml_predictor.trust.task_runner --root "' + $resolvedRoot + '"'
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) {
    $legacyRunner = Join-Path $resolvedRoot 'scripts\collect-next-race.ps1'
    $legacyArguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $legacyRunner + '" -ProjectRoot "' + $resolvedRoot + '"'
    $legacyAction = $existingTask.Actions.Count -eq 1 -and $existingTask.Actions.Execute -eq 'powershell.exe' -and $existingTask.Actions.Arguments -eq $legacyArguments
    $currentAction = $existingTask.Actions.Count -eq 1 -and $existingTask.Actions.Execute -eq $pythonPath -and $existingTask.Actions.Arguments -eq $arguments
    if ($existingTask.Actions.WorkingDirectory -ne $resolvedRoot -or (-not $legacyAction -and -not $currentAction)) {
        throw 'The task name is used by an unexpected action; inspect it before changing anything.'
    }
}
$action = New-ScheduledTaskAction -Execute $pythonPath -Argument $arguments -WorkingDirectory $resolvedRoot
$periodic = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 365)
$startup = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 4) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
if ($Preview) {
    [pscustomobject]@{
        TaskName = $taskName
        ExistingState = if ($existingTask) { $existingTask.State } else { 'Absent' }
        Execute = $pythonPath
        Arguments = $arguments
        LogonType = 'Password'
        RepetitionMinutes = 5
        StartupTrigger = $true
        WakeToRun = $true
    }
    return
}
if (-not $Credential) {
    $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    $Credential = Get-Credential -UserName $currentUser -Message 'Windows needs this account password to run the collector while signed out. Enter it in the local Windows prompt.'
}
if (-not $Credential) { throw 'Task registration requires a local Windows credential.' }
$password = $Credential.GetNetworkCredential().Password
try {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger @($periodic, $startup) -Settings $settings -User $Credential.UserName -Password $password -RunLevel Limited -Description 'Retain bounded immutable post-qualifying F1 captures. Raw and model artifacts remain local.' -Force | Select-Object TaskName, State
}
finally { $password = $null }
