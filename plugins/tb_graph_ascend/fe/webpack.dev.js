/*
 * -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  mode: 'development',
  devtool: 'eval-cheap-source-map',

  entry: {
    app: './src/index',
  },

  output: {
    filename: '[name].bundle.js',
    path: path.resolve(__dirname, 'dist'),
    publicPath: '/',
  },

  devServer: {
    static: {
      directory: path.join(__dirname, 'dist'),
    },
    compress: false,
    proxy: [
      {
        context: (pathname) => {
          return !pathname.match(/(\.js|\.css|\.html|\.ico|\.svg)$/);
        },
        target: 'http://127.0.0.1:6006',
        changeOrigin: true,
        secure: false,
        pathRewrite: {
          '^/(.*)': '/data/plugin/graph_ascend/$1',
        },
        on: {
          error: (err, req, res) => {
            if (res && !res.headersSent) {
              res.writeHead(500, { 'Content-Type': 'text/plain' });
              res.end('Proxy Error');
            }
          },
          proxyReqWs: (proxyReq, req, socket) => {
            socket.on('error', (error) => {
              console.error('WebSocket proxy error:', error);
            });
          },
        },
      },
    ],
    webSocketServer: {
      type: 'ws',
      options: {
        path: '/ws',
        noInfo: true,
      },
    },

    server: {
      type: 'http', // 可选: 'http' | 'https' | 'http2'
    },

    hot: true,
    liveReload: true,
    port: 8080,
    open: false,
    client: {
      overlay: {
        errors: true,
        warnings: false,
      },
    },
    headers: {
      'X-Proxy-Source': 'webpack-dev-server',
    },
  },

  module: {
    rules: [
      {
        test: /\.html$/,
        use: [
          {
            loader: 'html-loader',
            options: {
              sources: false,
            },
          },
        ],
      },
      {
        test: /\.ts?$/,
        use: {
          loader: 'ts-loader',
          options: {
            transpileOnly: true,
            experimentalWatchApi: true,
          },
        },
        exclude: /node_modules/,
      },
      {
        test: /\.css$/i,
        use: [
          'style-loader',
          {
            loader: 'css-loader',
            options: {
              sourceMap: true,
            },
          },
        ],
      },
    ],
  },

  resolve: {
    extensions: ['.ts', '.js', '.json'],
  },

  plugins: [
    new HtmlWebpackPlugin({
      inject: 'body',
      template: './index.html',
      minify: false,
    }),
  ],

  optimization: {
    removeAvailableModules: false,
    removeEmptyChunks: false,
    splitChunks: false,
  },
};
