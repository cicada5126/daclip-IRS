(function(){"use strict";const i="https://unpkg.com/@ffmpeg/core@0.12.1/dist/umd/ffmpeg-core.js";var o;(function(r){r.LOAD="LOAD",r.EXEC="EXEC",r.WRITE_FILE="WRITE_FILE",r.READ_FILE="READ_FILE",r.DELETE_FILE="DELETE_FILE",r.RENAME="RENAME",r.CREATE_DIR="CREATE_DIR",r.LIST_DIR="LIST_DIR",r.DELETE_DIR="DELETE_DIR",r.ERROR="ERROR",r.DOWNLOAD="DOWNLOAD",r.PROGRESS="PROGRESS",r.LOG="LOG",r.MOUNT="MOUNT",r.UNMOUNT="UNMOUNT"})(o||(o={}));const a=new Error("unknown message type"),f=new Error("ffmpeg is not loaded, call `await ffmpeg.load()` first"),u=new Error("failed to import ffmpeg-core.js");let e;const O=async({coreURL:r=i,wasmURL:n,workerURL:s})=>{const E=!e,t=r,c=n||r.replace(/.js$/g,".wasm"),U=s||r.replace(/.js$/g,".worker.js");try{importScripts(t)}catch{if(self.createFFmpegCore=(await import(t)).default,!self.createFFmpegCore)throw u}return e=await self.createFFmpegCore({mainScriptUrlOrBlob:`${t}#${btoa(JSON.stringify({wasmURL:c,workerURL:U}))}`}),e.setLogger(R=>self.postMessage({type:o.LOG,data:R})),e.setProgress(R=>self.postMessage({type:o.PROGRESS,data:R})),E},l=({args:r,timeout:n=-1})=>{e.setTimeout(n),e.exec(...r);const s=e.ret;return e.reset(),s},D=({path:r,data:n})=>(e.FS.writeFile(r,n),!0),L=({path:r,encoding:n})=>e.FS.readFile(r,{encoding:n}),m=({path:r})=>(e.FS.unlink(r),!0),S=({oldPath:r,newPath:n})=>(e.FS.rename(r,n),!0),I=({path:r})=>(e.FS.mkdir(r),!0),N=({path:r})=>{const n=e.FS.readdir(r),s=[];for(const E of n){const t=e.FS.stat(`${r}/${E}`),c=e.FS.isDir(t.mode);s.push({name:E,isDir:c})}return s},A=({path:r})=>(e.FS.rmdir(r),!0),w=({fsType:r,options:n,mountPoint:s})=>{let E=r,t=e.FS.filesystems[E];return t?(e.FS.mount(t,n,s),!0):!1},k=({mountPoint:r})=>(e.FS.unmount(r),!0);self.onmessage=async({data:{id:r,type:n,data:s}})=>{const E=[];let t;try{if(n!==o.LOAD&&!e)throw f;switch(n){case o.LOAD:t=await O(s);break;case o.EXEC:t=l(s);break;case o.WRITE_FILE:t=D(s);break;case o.READ_FILE:t=L(s);break;case o.DELETE_FILE:t=m(s);break;case o.RENAME:t=S(s);break;case o.CREATE_DIR:t=I(s);break;case o.LIST_DIR:t=N(s);break;case o.DELETE_DIR:t=A(s);break;case o.MOUNT:t=w(s);break;case o.UNMOUNT:t=k(s);break;default:throw a}}catch(c){self.postMessage({id:r,type:o.ERROR,data:c.toString()});return}t instanceof Uint8Array&&E.push(t.buffer),self.postMessage({id:r,type:n,data:t},E)}})();
//# sourceMappingURL=worker-DJ3jufjD.js.map
