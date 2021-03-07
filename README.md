# My personal website

The website was built with Jekyll and the *Minimal Mistakes* theme (by [Michael Rose](https://mademistakes.com)).

To test locally:
```shell script
bundle exec jekyll serve
# Run in the production mode if needed:
#   JEKYLL_ENV=production bundle exec jekyll serve
```

Add a submodule and a symlink pointing to the `Articles` repository 
([here](https://stackoverflow.com/a/27770463) and [here](https://stackoverflow.com/a/18712756))
```shell script
git submodule add -b master https://github.com/rolczynski/Articles
git mv Articles _posts
# and update:
git submodule update --rebase --remote
# and commit changes
git add _posts
```
