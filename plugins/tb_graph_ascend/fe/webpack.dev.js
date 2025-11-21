/* -------------------------------------------------------------------------
 Copyright (c) 2025, Huawei Technologies.
 All rights reserved.

 Licensed under the Apache License, Version 2.0  (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
--------------------------------------------------------------------------------------------*/
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
