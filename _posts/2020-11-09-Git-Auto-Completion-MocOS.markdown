---
layout:     post
title:      "Git Auto Completion in MacOS"
description: "How to do auto-completion for git used in macOS"
date:       2020-11-09 23:45:00
author:     "Yangyang Fu"
header-img: assets/img/posts/header-img/woman-typing-macbook.jpg

categories:
  - Experience/Git
---

In MacOS, the `git` might not be able to do auto-completion, which would waste user's a lot of time to type all the commands. This post aims to solve the mentioned issue.

## Check Shell Type

MacOS Catalina terminal has shifted from `bash` shell to `-zsh` shell. These two shells have different startup files. 

  - `./.bash_profile` for `bash` shell
  - `./.zprofile` for `-zsh` shell

These two files can be used to execute startup commands such as configuring software environmental path when the shell is called upon.

## Check if auto-completion command is installed

The auto-completion needs to be installed so that the shell can call it. A typical installation can be checked using:

    ls /Library/Developer/CommandLineTools/usr/share/git-core

if successfully installed, you should see somehthing like this:

![1](figs/2020-11-09/1.png)

If not, try to install auto-completion by simplying copying and pasting a `git-completion.bash` or `git-completion.zsh` from internet.

## Add to Environmental Path

For `-zsh` shell, add the following command to the line of `./.zprofile`.

    autoload -Uz compinit && compinit

Restart the shell, you should be able to do auto-completion in `-zsh` shell now.

Similar methods can be used for `bash` shell.