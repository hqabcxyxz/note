[toc]

#### cheatsheet

#####  显示当前使用配置文件的路径

```shell
:echo $MYVIMRC
```

##### 关于选项值切换和查看

使用`set 选项值` 之后添加`!`可以切换选项值(选项值必须是布尔值),使用`?`可以显示当前选项.

##### 使用`<buffer>`可以对map和iabbrev设置限定本地缓冲区.







#### 建议

1. 使用映射时,任何时候都应该优先考虑noremap
2. 编写插件设置leader时,应使用`<localleader>`
3. 使用自定命令时,建议同时使用`BufRead`和`BufNewFile`

