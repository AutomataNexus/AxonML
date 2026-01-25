// PM2 Ecosystem Configuration for AxonML
// https://pm2.keymetrics.io/docs/usage/application-declaration/
//
// Usage:
//   pm2 start ecosystem.config.js           # Start all services
//   pm2 start ecosystem.config.js --only axonml-server     # Server only
//   pm2 start ecosystem.config.js --only axonml-dashboard  # Dashboard dev server only
//   pm2 stop axonml-server                  # Stop server
//   pm2 restart axonml-server               # Restart server
//   pm2 logs axonml-server                  # View logs
//   pm2 save                                # Save current process list
//   pm2 startup                             # Generate startup script (persist on reboot)
//
// First-time setup:
//   1. Build the release binary: cargo build --release -p axonml-server
//   2. Build dashboard: cd crates/axonml-dashboard && trunk build --release
//   3. Initialize database: ./AxonML_DB_Init.sh --with-user
//   4. Start with PM2: pm2 start ecosystem.config.js
//   5. Save process list: pm2 save
//   6. Enable startup: pm2 startup (follow instructions)
//
// Ports:
//   - axonml-server:    3021 (API backend)
//   - axonml-dashboard: 8082 (nginx serves static WASM, proxies API)
//   - trunk serve:      8081 (dev mode with hot reload)

module.exports = {
  apps: [
    {
      name: 'axonml-server',
      script: './target/release/axonml-server',
      args: '--port 3021',
      cwd: '/opt/AxonML',
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '2G',
      env: {
        RUST_LOG: 'info',
        RUST_BACKTRACE: '1',
      },
      env_production: {
        RUST_LOG: 'warn',
        RUST_BACKTRACE: '0',
      },
      // Graceful shutdown
      kill_timeout: 5000,
      wait_ready: true,
      listen_timeout: 10000,
      // Logging
      error_file: '/var/log/axonml/server-error.log',
      out_file: '/var/log/axonml/server-out.log',
      log_file: '/var/log/axonml/server-combined.log',
      time: true,
      // Restart policy
      exp_backoff_restart_delay: 100,
      max_restarts: 10,
      restart_delay: 1000,
    },
    {
      // Dashboard with trunk serve (static files + API proxy)
      name: 'axonml-dashboard',
      script: 'trunk',
      args: 'serve --port 8083',
      cwd: '/opt/AxonML/crates/axonml-dashboard',
      interpreter: 'none',
      autorestart: true,
      watch: false,
      // Logging
      error_file: '/var/log/axonml/dashboard-error.log',
      out_file: '/var/log/axonml/dashboard-out.log',
      log_file: '/var/log/axonml/dashboard-combined.log',
      time: true,
      // Restart policy
      max_restarts: 5,
      restart_delay: 3000,
    },
  ],
};
