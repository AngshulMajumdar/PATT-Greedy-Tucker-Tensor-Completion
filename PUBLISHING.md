# Publish the flat PATT repository on GitHub

The downloadable repository is deliberately **flat**: no subdirectories, no hidden dotfiles, and no Git metadata are included in the ZIP.

1. Open GitHub and create an empty repository, for example **PATT-Greedy-Tucker-Rank-Reallocation**. Do not initialize with an additional README, license, or `.gitignore` because `README.md` is already supplied.
2. Extract the ZIP and upload **the files from inside it** to the repository root, not the ZIP as one object. GitHub then renders `README.md` and all relative `chart_*.png` graphics automatically.
3. For local git users, `git init`, `git add .`, and `git push` work as usual. Git creates a local `.git` metadata folder, but **that folder is not part of the tracked repository contents or this archive**.
4. The link is not live until you create and publish the GitHub repository. This package does not claim a public GitHub URL.

To reproduce the published numerical tables and images:

```bash
python -m pip install -r requirements.txt
python reproduce_summary.py
python plot_results.py
```

The 72 `*.mat` files are measured benchmark inputs. The main CSV/JSONL files are complete numerical records. The root-level manuscript and LaTeX files are included so the README connects directly to the research article.
