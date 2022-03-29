> First Update: 29, March, 2022

## Introduction

When you using `git` on server, sometimes you will occasionally encounter problems such as:

```
~ :> git push origin my-branch
Username for 'https://github.com': myusername
Password for 'https://myusername@github.com': mypassword
remote: Invalid username or password.
fatal: Authentication failed for 'https://github.com/my-repository’
```

As of fall 2021, GitHub will no longer allow usage of a password alone. One good option is to use a [personal access token](https://docs.github.com/en/github/authenticating-to-github/creating-a-personal-access-token) in place of a password [^1].

## Solution

### Reset your credential
If you are using IDEs such as Jetbrains, they may save you previous login details, so you have to forget or set the `user.name` and `user.email` again:

```
# forget your previous auth info
git config --system --unset credential.helper

# set user.name and user.email globally
git config --global user.name [username]
git config --global user.email [email]
```

You should substitute your own `username` and `email` in these two field: `[username]` and `[email]`.

### Generate your access token

First you need to login your Github account in a browser. After that, go to [Personal access token](https://github.com/settings/tokens) [^2].

> You can also access this area by steps: 
> ```
> Github -> Settings -> Developer Settings -> Personal access tokens
> ```

Click on `Generate new token`:

![personal_access_token.png](../../_media/personal_access_token.png)

After an authentication, you will get the `New personal access token` area. First, you need to add `Note`, basically a tag which can make you memorise when you will use this token. You can set the `Expiration`  to `No expiration` for convenience. After that, select the necessary permissions. Usually, a tick on `repo` will do most of the work.

![access_token_gen.png](../../_media/access_token_gen.png)

**⚠️ Warning: The token will only appear once! So you'd better write it down or save it with password tools.**

After all that, you can paste the Personal Access Token into the “Password” field when you authenticate via the command line.

### Saving your password or token to avoid entering it 

You can [save, or cache your credentials](https://docs.github.com/en/get-started/getting-started-with-git/caching-your-github-credentials-in-git) so that you don't have to reenter them each time you interact with the remote repository. [^3]

#### GitHub Cli

One possible way is to use `GitHub Cli` as an official tool to store your Git credentials for you.

1. Install GitHub CLI on macOS, Windows, or Linux.
2. In the command line, enter `gh auth login`, then follow the prompts.
	1. When prompted for your preferred protocol for Git operations, select `HTTPS`.
	2. When asked if you would like to authenticate to Git with your GitHub credentials, enter `Y`.

You can simply install `Git for Windows`, which includes Git Credential Manager (GCM) to store your credentials and connect to GitHub over HTTPs.

####  Native Credential Tools

Your credentials can be stored in the keychain of your operating system or cached in memory or in a file.

To cache in memory, in the MacOS keychain, or in the Windows keychain, choose the relevant one of these three invocations:

```
# in memory:
git config --global credential.helper cache
# MacOS
git config --global credential.helper osxkeychain
# Windows
git config --global credential.helper wincred
```

If you don't want your in memory cache expires too soon, you can set a time for it:

```
git config --global credential.helper 'cache --timeout=6480000'
```

If you prefer to set the credential helper on a repository-specific basis, you can omit the `--global` flag.

### Using SSH keys

There are many tutorials on putting your SSH public key in your GitHub account. I will say no more about it.


## Reference

[^1]: [About authentication to GitHub: Authenticating with the command line](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github#authenticating-with-the-command-line)

[^2]: [Creating a personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token)

[^3]: [Authenticating to Remote Git Repositories](https://statistics.berkeley.edu/computing/faqs/git-auth)