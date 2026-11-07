# 🚀 Git Repository Stats

A beautiful Python tool for displaying comprehensive git repository statistics with emoji indicators and tabular formatting.

## 📋 Features

- **📊 Repository Overview**: Branch, commit hash, author, timestamp, and commit count
- **📂 File Status Tracking**: Detailed information about staged, modified, deleted, and conflicted files
- **🎨 Beautiful Formatting**: Uses `tabulate` library for professional table displays
- **😊 Emoji Indicators**: Visual status indicators for quick understanding
- **🔧 Flexible Options**: Multiple display formats and customization options

## 🛠️ Installation

```bash
pip install tabulate
```

## 📖 Usage Examples

### 1. Basic Usage - Default View

```python
from xcube.utils import print_git_stats

# Shows repository info + file details (excludes untracked files by default)
print_git_stats()
```

**Output:**
```
🚀 Git Repository Statistics 🚀
================================================================================
📋 Property        📝 Value
-----------------  ----------------------------------------------------------
📁 Repository      my-awesome-project
🌿 Branch          main
🔍 Commit Hash     a1b2c3d (a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0)
💬 Last Message    Add new feature implementation
👤 Author          John Doe <john@example.com>
📅 Date & Time     2025-06-12 10:30:45 -0400
📊 Total Commits   127
📋 Working Directory  ⚡ MIXED (3 staged, 2 unstaged)
🔬 Inspect Commit  git show a1b2c3d --stat
🕰️ Checkout Commit git checkout a1b2c3d
```

### 2. Include Untracked Files

```python
# Show untracked files as well
print_git_stats(show_untracked=True)
```

### 3. Basic Info Only

```python
# Repository info without file details
print_git_stats(show_files=False)
```

### 4. Different Table Formats

```python
# Available formats: grid, fancy_grid, simple, plain, pipe, orgtbl
print_git_stats(table_format="fancy_grid")
print_git_stats(table_format="simple")
```

### 5. Compact One-Line Summary

```python
from xcube.utils import get_compact_git_info

print(get_compact_git_info())
# Output: 📁 my-project | 🌿 main | 🔍 a1b2c3d | 👤 John Doe | ⚡ MIXED (3 staged, 2 unstaged) | 🔬 git show a1b2c3d
```

### 6. Get Raw Data

```python
from xcube.utils import get_git_repo_stats
import json

stats = get_git_repo_stats()
if stats:
    print(json.dumps(stats, indent=2))
```

## 🔬 Git Commands for Reproducibility

The tool provides ready-to-use git commands for reproducing the exact state:

### 🔍 Inspect the Commit
```bash
git show a1b2c3d --stat
```
Shows the commit details, changes, and file statistics.

### 🕰️ Checkout the Commit  
```bash
git checkout a1b2c3d
```
Switches to the exact commit state (detached HEAD mode).

### 🔄 Additional Useful Commands
```bash
# See full commit details with diff
git show a1b2c3d

# See commit in context with surrounding commits  
git log --oneline -n 10 a1b2c3d

# Create a branch from this commit
git checkout -b experiment-reproduction a1b2c3d

# Compare current state with this commit
git diff a1b2c3d
```

## 📊 Status Indicators

### 🏷️ Working Directory Status
- **✅ CLEAN** - No changes to commit
- **🟢 STAGED (n)** - Files staged and ready to commit
- **🟡 CHANGES (n)** - Unstaged changes or untracked files
- **⚡ MIXED (n staged, n unstaged)** - Both staged and unstaged changes
- **🔥 CONFLICTS (n)** - Merge conflicts that need resolution

### 📄 Individual File Status
- **🆕 New (Staged)** - New files added to staging area
- **✏️ Modified (Staged)** - Modified files in staging area
- **🗑️ Deleted (Staged)** - Deleted files in staging area
- **📝 Renamed (Staged)** - Renamed files in staging area
- **⚠️ Modified** - Modified files not yet staged
- **❌ Deleted** - Deleted files not yet staged
- **❓ Untracked** - New files not tracked by git (optional)
- **🔥 Conflict** - Files with merge conflicts

## 🎛️ Function Parameters

### `print_git_stats(table_format="grid", show_files=True, show_untracked=False)`

- **`table_format`**: Table style for output
  - Options: `"grid"`, `"fancy_grid"`, `"simple"`, `"plain"`, `"pipe"`, `"orgtbl"`
  - Default: `"grid"`

- **`show_files`**: Whether to display detailed file status
  - Default: `True`

- **`show_untracked`**: Whether to include untracked files in output
  - Default: `False`

## 🚀 Real-World Examples

### Clean Repository
```python
print_git_stats()
```
Shows: `📋 Working Directory: ✅ CLEAN`

### Development in Progress
```python
print_git_stats()
```
Shows staged changes, modified files, but hides untracked temp files by default.

### Code Review Ready
```python
print_git_stats(show_files=False)
```
Quick overview without file details - perfect for status checks.

### Full Repository Audit
```python
print_git_stats(show_untracked=True)
```
Complete view including all untracked files.

## 🔧 Advanced Usage

### Custom Status Check Script
```python
#!/usr/bin/env python3
from xcube.utils import get_git_repo_stats, get_compact_git_info

def check_repo_status():
    stats = get_git_repo_stats()
    if not stats:
        print("❌ Not in a git repository")
        return False
    
    file_states = stats['file_states']
    has_staged = any([
        file_states['staged_new'],
        file_states['staged_modified'],
        file_states['staged_deleted'],
        file_states['staged_renamed']
    ])
    
    if has_staged:
        print("🟢 Ready to commit!")
        print(get_compact_git_info())
        return True
    else:
        print("🟡 No changes staged for commit")
        return False

if __name__ == "__main__":
    check_repo_status()
```

### Integration with CI/CD
```python
# In your build script
stats = get_git_repo_stats()
if stats:
    commit_hash = stats['short_hash']
    print(f"Building commit: {commit_hash}")
```

## 🎨 Table Format Examples

### Grid (Default)
```
+------------------+----------------------------------------+
| 📋 Property      | 📝 Value                               |
+==================+========================================+
| 📁 Repository    | my-project                             |
+------------------+----------------------------------------+
```

### Fancy Grid
```
╒══════════════════╤════════════════════════════════════════╕
│ 📋 Property      │ 📝 Value                               │
╞══════════════════╪════════════════════════════════════════╡
│ 📁 Repository    │ my-project                             │
╘══════════════════╧════════════════════════════════════════╛
```

### Simple
```
📋 Property        📝 Value
-----------------  ----------------------------------------
📁 Repository      my-project
```

## 🤝 Contributing

This tool is designed to be lightweight and focused. Feel free to extend it for your specific needs!

## 📄 License

Free to use and modify for your projects.

---

**Happy coding! 🎉**