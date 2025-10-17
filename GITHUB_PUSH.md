# how to push to github

## step 1: create repo on github

go to github.com and:
1. click the + button (top right)
2. click "new repository"
3. name it something like `badminton-speed-analyzer` or whatever
4. **dont** initialize with readme (we already have one)
5. click create

## step 2: run these commands

```bash
# initialize git
git init

# add all files
git add .

# commit
git commit -m "initial commit: badminton speed analyzer with pytorch backend"

# add your github repo as remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# push to github
git branch -M main
git push -u origin main
```

## full example (replace with your info)

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/yourusername/badminton-speed-analyzer.git
git branch -M main
git push -u origin main
```

## what gets pushed

- all code (frontend + backend)
- documentation (README, QUICKSTART, etc)
- docker files
- tests
- scss source files

## what DOESNT get pushed (gitignore)

- virtual env (venv/)
- python cache (__pycache__)
- video files (*.mp4, etc) - too large
- yolo models (*.pt) - too large, downloads automatically
- logs
- .env files (secrets)
- user uploads/results
- node_modules

## after pushing

your repo will be at:
```
https://github.com/YOUR_USERNAME/REPO_NAME
```

## updating later

```bash
# after making changes
git add .
git commit -m "describe what you changed"
git push
```

## if you want to add large files later

github has a 100mb file limit. if you trained a custom yolo model and want to share it, use git lfs:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git commit -m "add lfs tracking"
```

or just link to it from google drive/dropbox in the readme.

## making it public/private

you can change this in github repo settings → danger zone → change visibility

## adding collaborators

settings → collaborators → add people

## common issues

**"remote origin already exists"**
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

**"failed to push"**
- make sure you created the repo on github first
- check you have the right repo url
- try using ssh instead of https: `git@github.com:YOUR_USERNAME/REPO_NAME.git`

**"authentication failed"**
- use personal access token instead of password
- or set up ssh keys (github.com/settings/keys)

thats it. your code is now backed up and shareable.

