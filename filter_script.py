from git_filter_repo import FilterRepo, Commit

def commit_callback(commit: Commit):
    if commit.author_email == b"tiz.labruna@gmail.com":
        commit.author_name = b"mwozgpt"
        commit.author_email = b"ugorenato0@gmail.com"
    if commit.committer_email == b"tiz.labruna@gmail.com":
        commit.committer_name = b"mwozgpt"
        commit.committer_email = b"ugorenato0@gmail.com"

filter_repo = FilterRepo(commit_callback=commit_callback)
filter_repo.run()
