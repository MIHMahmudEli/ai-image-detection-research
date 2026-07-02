# DALL-E 3 download runner
# Runs the Python script as a background job so it won't time out.
param(
    [int]$Start = 0,
    [int]$Batch = 3
)

$NUM_FILES = 67
$logFile = "$PSScriptRoot\dalle3_progress.txt"

for ($i = $Start; $i -lt $NUM_FILES; $i += $Batch) {
    $end = [Math]::Min($i + $Batch, $NUM_FILES)
    Write-Output "[$(Get-Date -Format HH:mm:ss)] Processing files $i to $($end-1)..."

    $result = python "$PSScriptRoot\08_download_dalle3.py" $i $end 2>&1
    $result | Out-File -FilePath $logFile -Append

    $imgCount = (Get-ChildItem "dataset\images\DALL-E3" -File | Measure-Object).Count
    Write-Output "[$(Get-Date -Format HH:mm:ss)] DALL-E3 images so far: $imgCount"
    "$(Get-Date -Format yyyy-MM-dd HH:mm:ss) processed $i-$($end-1), total images: $imgCount" | Out-File -FilePath $logFile -Append
}

Write-Output "Done! Total DALL-E3 images: $((Get-ChildItem 'dataset\images\DALL-E3' -File | Measure-Object).Count)"
