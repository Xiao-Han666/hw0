aloha
1、github的常见使用方法，包括"git switch -c <new branch>"创建并切换新分支，新分支会继承当前分支的所有提交，用"git merge <branch>"把分支合并进当前分支。
2、"git checkout <hash>"可以回到某个旧提交，此时处于 detached HEAD 状态，"git log"可以查看历史记录，添加参数"--all"可以查看所有的历史记录等，进一步对于github的多人开发和合作有了了解。
3、hugginface和transformer的基本语法，可以使用"ResNetForImageClassification.from_pretrained("microsoft/resnet-18")"加载预训练 ResNet，对于更多的其他模型可以类似地载入。进一步地，了解了模型的基本用法和训练模型的相关基本流程。