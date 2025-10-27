# GitHub Upload Instructions

Follow these steps to upload your modified AgentFormer codebase to your personal GitHub account.

## Prerequisites

- GitHub account
- Git installed and configured with your credentials

## Option 1: Create New Repository (Recommended)

This creates a fresh repository under your GitHub account.

### Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Fill in:
   - **Repository name**: `vision-augmented-agentformer` (or your preferred name)
   - **Description**: "AgentFormer with BEVDepth visual features for trajectory prediction"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
3. Click "Create repository"

### Step 2: Update Git Remote

```bash
# Remove current remote (points to original AgentFormer)
git remote remove origin

# Add your new repository (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/vision-augmented-agentformer.git

# Verify
git remote -v
```

### Step 3: Commit and Push

```bash
# Commit all staged changes
git commit -m "Add BEVDepth integration and dataset filtering for nuScenes

Major changes:
- Added conditional BEVDepth dataloader with use_bev flag
- Implemented dataset filtering for 10% nuScenes subset (85 scenes)
- Fixed training pipeline bugs (variable naming, forward pass, logging)
- Added quick pipeline test script for fast verification
- Created comprehensive documentation (README_MODIFICATIONS.md, SETUP_GUIDE.md)
- Updated .gitignore to exclude large dataset files

This enables ablation study comparing baseline AgentFormer vs vision-augmented AgentFormer.

🤖 Generated with Claude Code https://claude.com/claude-code

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to your repository
git push -u origin main
```

### Step 4: Verify Upload

1. Go to `https://github.com/YOUR_USERNAME/vision-augmented-agentformer`
2. Verify files are uploaded
3. Check that README_MODIFICATIONS.md displays correctly

## Option 2: Fork Original Repository

This maintains connection to original AgentFormer.

### Step 1: Fork on GitHub

1. Go to https://github.com/Khrylx/AgentFormer
2. Click "Fork" button in top right
3. Select your account as destination

### Step 2: Update Remote

```bash
# Rename current remote to 'upstream'
git remote rename origin upstream

# Add your fork as 'origin' (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/AgentFormer.git

# Verify
git remote -v
```

Should show:
```
origin    https://github.com/YOUR_USERNAME/AgentFormer.git (fetch)
origin    https://github.com/YOUR_USERNAME/AgentFormer.git (push)
upstream  https://github.com/Khrylx/AgentFormer.git (fetch)
upstream  https://github.com/Khrylx/AgentFormer.git (push)
```

### Step 3: Commit and Push

```bash
# Commit changes (same as Option 1)
git commit -m "Add BEVDepth integration and dataset filtering for nuScenes

Major changes:
- Added conditional BEVDepth dataloader with use_bev flag
- Implemented dataset filtering for 10% nuScenes subset (85 scenes)
- Fixed training pipeline bugs (variable naming, forward pass, logging)
- Added quick pipeline test script for fast verification
- Created comprehensive documentation (README_MODIFICATIONS.md, SETUP_GUIDE.md)
- Updated .gitignore to exclude large dataset files

This enables ablation study comparing baseline AgentFormer vs vision-augmented AgentFormer.

🤖 Generated with Claude Code https://claude.com/claude-code

Co-Authored-By: Claude <noreply@anthropic.com>"

# Push to your fork
git push -u origin main
```

## After Pushing

### Update Repository Settings

1. Go to repository settings on GitHub
2. Under "About" (top right), add:
   - **Description**: "AgentFormer with BEVDepth visual features for trajectory prediction on nuScenes"
   - **Topics**: `trajectory-prediction`, `autonomous-driving`, `nuscenes`, `transformer`, `bevdepth`, `computer-vision`
   - **Website** (optional): Link to paper or project page

### Add Main README

GitHub will display README_MODIFICATIONS.md if you rename it to README.md, or you can create a new README.md:

```bash
# Option A: Rename (recommended to keep both)
cp README_MODIFICATIONS.md README.md
git add README.md
git commit -m "Add main README for GitHub display"
git push
```

### Create Release (Optional)

After confirming everything works:

1. Go to repository → Releases → "Create a new release"
2. Tag version: `v1.0.0`
3. Release title: "Initial Release - BEVDepth Integration"
4. Description:
   ```
   Initial release of Vision-Augmented AgentFormer

   Features:
   - Conditional BEVDepth integration for visual features
   - Dataset filtering for 10% nuScenes subset
   - Quick pipeline test for verification
   - Complete setup documentation

   Ready for ablation study comparing baseline vs vision-augmented trajectory prediction.
   ```
5. Click "Publish release"

## Troubleshooting

### Authentication Required

If you see "Authentication failed":

```bash
# Option 1: Use GitHub token (recommended)
# 1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
# 2. Generate new token with 'repo' scope
# 3. Use token as password when pushing

# Option 2: Use SSH
# 1. Set up SSH keys: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
# 2. Update remote URL:
git remote set-url origin git@github.com:YOUR_USERNAME/vision-augmented-agentformer.git
```

### Large Files Warning

If Git warns about large files:

```bash
# Check file sizes
git ls-files -z | xargs -0 du -h | sort -h | tail -20

# Remove large files from staging
git reset HEAD path/to/large/file

# Add to .gitignore if not already there
echo "path/to/large/file" >> .gitignore
```

### Push Rejected

If push is rejected due to conflicts:

```bash
# Pull changes first
git pull origin main --rebase

# Resolve any conflicts, then push
git push origin main
```

## Clone on New Machine

After uploading, to clone on a new machine:

```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/vision-augmented-agentformer.git
cd vision-augmented-agentformer

# Follow SETUP_GUIDE.md for installation
```

## Summary of What Gets Uploaded

### Included (in repository):
- ✅ Source code (.py files)
- ✅ Configuration files (.yml)
- ✅ Documentation (.md files)
- ✅ Scripts (scripts/)
- ✅ Requirements (requirements.txt)

### Excluded (via .gitignore):
- ❌ Dataset files (datasets/, data/*.pkl) - too large
- ❌ Trained models (results/) - too large
- ❌ Build artifacts (build/, *.egg-info/)
- ❌ Python cache (__pycache__/, *.pyc)
- ❌ Logs (*.log)

Users will need to download the nuScenes dataset separately (see SETUP_GUIDE.md).

## Repository Size

Your repository should be:
- **Code only**: ~5-10 MB
- **With all commits**: ~20-30 MB

If it's larger (>100 MB), you likely have dataset files committed. Run:

```bash
# Check what's taking space
du -sh .git/objects/pack/*

# If dataset files slipped in, remove from history:
git filter-branch --tree-filter 'rm -rf datasets/' HEAD
```

## Next Steps

After uploading:
1. Share repository URL with collaborators
2. Set up GitHub Actions for CI/CD (optional)
3. Enable GitHub Pages for documentation (optional)
4. Add badges to README (build status, license, etc.)

Your code is now reproducible and shareable!
