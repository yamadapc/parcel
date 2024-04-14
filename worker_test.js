const {isMainThread, Worker} = require('worker_threads');

const seri = '';

if (!isMainThread) {
  console.log('Worker thread');
  this.onmessage = data => {
    console.log(data);
  };
  return;
}

const worker = new Worker(__filename);
