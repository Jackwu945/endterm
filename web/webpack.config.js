const path = require('path');

module.exports = {
    // 定义入口文件
    entry: './js/main.js',
    // 定义输出
    output: {
        filename: 'bundle.js',
        path: path.resolve(__dirname, 'dist')
    },
    // 配置模块解析方式
    resolve: {
        // 设置别名
        alias: {
            // 这里假设node_modules在项目根目录下
            'dplayer': path.resolve(__dirname, 'node_modules/dplayer/dist/DPlayer.min.js')
        }
    },
    // 配置模块如何解析
    module: {
        rules: [
            // 添加其他规则
        ]
    },
    // 添加插件
    plugins: [
        // 添加其他插件
    ]
};