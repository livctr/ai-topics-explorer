# Why?
# Cannot figure out rsync on Windows or WSL.. getting some OpenSSL issues

# Usage:
# .\scp_copy.ps1 -RemoteDestination "user@host:/path/on/remote"

param(
    [Parameter(Mandatory = $true,
        HelpMessage = "Full remote destination in the format user@host:/path/to/destination")]
    [string]$RemoteDestination
)

# Define names to exclude (applies at every level)
$excludes = @(".git", "node_modules")

# Build a temporary directory path (we’ll copy files here)
$tempDir = Join-Path $env:TEMP "scp_temp_copy"

# Remove any previous temporary copy
if (Test-Path $tempDir) {
    Remove-Item $tempDir -Recurse -Force
}
New-Item -ItemType Directory -Path $tempDir | Out-Null

# Recursive function to copy items from $Source to $Destination,
# skipping any item whose name is in the $excludes array.
function Copy-ItemExcluding {
    param(
        [string]$Source,
        [string]$Destination
    )
    Get-ChildItem -Path $Source -Force | Where-Object { ($excludes -notcontains $_.Name) -and ($_.Extension -ne ".pyc") } | ForEach-Object {
        $destPath = Join-Path $Destination $_.Name
        if ($_.PSIsContainer) {
            New-Item -ItemType Directory -Path $destPath -Force | Out-Null
            Copy-ItemExcluding -Source $_.FullName -Destination $destPath
        } else {
            Copy-Item $_.FullName $destPath -Force
        }
    }
}

# Determine the source directory:
# Since this script now resides in the "scripts" directory, we want to use its parent directory (the project root).
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$sourceDir = Split-Path -Parent $scriptDir
Write-Output "Copying files from source directory: $sourceDir"

# Copy everything from the source directory (excluding .git and node_modules)
Copy-ItemExcluding -Source $sourceDir -Destination $tempDir

# Gather all files and directories inside the temporary folder.
# We use Get-ChildItem on $tempDir so that we don’t include the temp folder itself.
$itemsToCopy = Get-ChildItem -Path $tempDir -Force | ForEach-Object { $_.FullName }

# Use scp to recursively copy the contents to the remote destination.
# This transfers the items from $tempDir into the remote folder without nesting them in "scp_temp_copy".
scp -r -i ~/.ssh/cs-topics-explorer-server-key.pem $itemsToCopy $RemoteDestination
