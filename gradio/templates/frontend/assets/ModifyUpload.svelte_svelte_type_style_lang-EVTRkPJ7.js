import{p as Te}from"./index-D6iiusuW.js";import"./Index-DvJ399W-.js";const{SvelteComponent:Se,append:y,attr:S,detach:ne,element:U,flush:M,init:Ue,insert:ie,noop:x,safe_not_equal:We,set_data:J,set_style:K,space:Q,text:j,toggle_class:$}=window.__gradio__svelte__internal,{onMount:je,createEventDispatcher:Ce,onDestroy:qe}=window.__gradio__svelte__internal;function ee(t){let e,l,n,r,o=N(t[2])+"",d,f,s,a,m=t[2].orig_name+"",_;return{c(){e=U("div"),l=U("span"),n=U("div"),r=U("progress"),d=j(o),s=Q(),a=U("span"),_=j(m),K(r,"visibility","hidden"),K(r,"height","0"),K(r,"width","0"),r.value=f=N(t[2]),S(r,"max","100"),S(r,"class","svelte-1vsfomn"),S(n,"class","progress-bar svelte-1vsfomn"),S(a,"class","file-name svelte-1vsfomn"),S(e,"class","file svelte-1vsfomn")},m(h,c){ie(h,e,c),y(e,l),y(l,n),y(n,r),y(r,d),y(e,s),y(e,a),y(a,_)},p(h,c){c&4&&o!==(o=N(h[2])+"")&&J(d,o),c&4&&f!==(f=N(h[2]))&&(r.value=f),c&4&&m!==(m=h[2].orig_name+"")&&J(_,m)},d(h){h&&ne(e)}}}function Ne(t){let e,l,n,r=t[0].length+"",o,d,f=t[0].length>1?"files":"file",s,a,m,_=t[2]&&ee(t);return{c(){e=U("div"),l=U("span"),n=j("Uploading "),o=j(r),d=Q(),s=j(f),a=j("..."),m=Q(),_&&_.c(),S(l,"class","uploading svelte-1vsfomn"),S(e,"class","wrap svelte-1vsfomn"),$(e,"progress",t[1])},m(h,c){ie(h,e,c),y(e,l),y(l,n),y(l,o),y(l,d),y(l,s),y(l,a),y(e,m),_&&_.m(e,null)},p(h,[c]){c&1&&r!==(r=h[0].length+"")&&J(o,r),c&1&&f!==(f=h[0].length>1?"files":"file")&&J(s,f),h[2]?_?_.p(h,c):(_=ee(h),_.c(),_.m(e,null)):_&&(_.d(1),_=null),c&2&&$(e,"progress",h[1])},i:x,o:x,d(h){h&&ne(e),_&&_.d()}}}function N(t){return t.progress*100/(t.size||0)||0}function Ie(t){let e=0;return t.forEach(l=>{e+=N(l)}),document.documentElement.style.setProperty("--upload-progress-width",(e/t.length).toFixed(2)+"%"),e/t.length}function Me(t,e,l){let{upload_id:n}=e,{root:r}=e,{files:o}=e,{stream_handler:d}=e,f,s=!1,a,m,_=o.map(u=>({...u,progress:0}));const h=Ce();function c(u,g){l(0,_=_.map(z=>(z.orig_name===u&&(z.progress+=g),z)))}return je(()=>{if(f=d(new URL(`${r}/upload_progress?upload_id=${n}`)),f==null)throw new Error("Event source is not defined");f.onmessage=async function(u){const g=JSON.parse(u.data);s||l(1,s=!0),g.msg==="done"?(f?.close(),h("done")):(l(7,a=g),c(g.orig_name,g.chunk_size))}}),qe(()=>{(f!=null||f!=null)&&f.close()}),t.$$set=u=>{"upload_id"in u&&l(3,n=u.upload_id),"root"in u&&l(4,r=u.root),"files"in u&&l(5,o=u.files),"stream_handler"in u&&l(6,d=u.stream_handler)},t.$$.update=()=>{t.$$.dirty&1&&Ie(_),t.$$.dirty&129&&l(2,m=a||_[0])},[_,s,m,n,r,o,d,a]}class Je extends Se{constructor(e){super(),Ue(this,e,Me,Ne,We,{upload_id:3,root:4,files:5,stream_handler:6})}get upload_id(){return this.$$.ctx[3]}set upload_id(e){this.$$set({upload_id:e}),M()}get root(){return this.$$.ctx[4]}set root(e){this.$$set({root:e}),M()}get files(){return this.$$.ctx[5]}set files(e){this.$$set({files:e}),M()}get stream_handler(){return this.$$.ctx[6]}set stream_handler(e){this.$$set({stream_handler:e}),M()}}const{SvelteComponent:Le,append:te,attr:k,binding_callbacks:Oe,bubble:D,check_outros:re,create_component:Re,create_slot:se,destroy_component:Be,detach:L,element:V,empty:oe,flush:w,get_all_dirty_from_scope:ae,get_slot_changes:ue,group_outros:fe,init:Ge,insert:O,listen:v,mount_component:He,prevent_default:P,run_all:Ke,safe_not_equal:Qe,set_style:de,space:Ve,stop_propagation:T,toggle_class:b,transition_in:E,transition_out:W,update_slot_base:_e}=window.__gradio__svelte__internal,{createEventDispatcher:Xe,tick:Ye,getContext:rt}=window.__gradio__svelte__internal;function Ze(t){let e,l,n,r,o,d,f,s,a,m,_;const h=t[26].default,c=se(h,t,t[25],null);return{c(){e=V("button"),c&&c.c(),l=Ve(),n=V("input"),k(n,"aria-label","file upload"),k(n,"data-testid","file-upload"),k(n,"type","file"),k(n,"accept",r=t[16]||void 0),n.multiple=o=t[6]==="multiple"||void 0,k(n,"webkitdirectory",d=t[6]==="directory"||void 0),k(n,"mozdirectory",f=t[6]==="directory"||void 0),k(n,"class","svelte-j5bxrl"),k(e,"tabindex",s=t[9]?-1:0),k(e,"class","svelte-j5bxrl"),b(e,"hidden",t[9]),b(e,"center",t[4]),b(e,"boundedheight",t[3]),b(e,"flex",t[5]),b(e,"disable_click",t[7]),de(e,"height","100%")},m(u,g){O(u,e,g),c&&c.m(e,null),te(e,l),te(e,n),t[34](n),a=!0,m||(_=[v(n,"change",t[18]),v(e,"drag",T(P(t[27]))),v(e,"dragstart",T(P(t[28]))),v(e,"dragend",T(P(t[29]))),v(e,"dragover",T(P(t[30]))),v(e,"dragenter",T(P(t[31]))),v(e,"dragleave",T(P(t[32]))),v(e,"drop",T(P(t[33]))),v(e,"click",t[13]),v(e,"drop",t[19]),v(e,"dragenter",t[17]),v(e,"dragleave",t[17])],m=!0)},p(u,g){c&&c.p&&(!a||g[0]&33554432)&&_e(c,h,u,u[25],a?ue(h,u[25],g,null):ae(u[25]),null),(!a||g[0]&65536&&r!==(r=u[16]||void 0))&&k(n,"accept",r),(!a||g[0]&64&&o!==(o=u[6]==="multiple"||void 0))&&(n.multiple=o),(!a||g[0]&64&&d!==(d=u[6]==="directory"||void 0))&&k(n,"webkitdirectory",d),(!a||g[0]&64&&f!==(f=u[6]==="directory"||void 0))&&k(n,"mozdirectory",f),(!a||g[0]&512&&s!==(s=u[9]?-1:0))&&k(e,"tabindex",s),(!a||g[0]&512)&&b(e,"hidden",u[9]),(!a||g[0]&16)&&b(e,"center",u[4]),(!a||g[0]&8)&&b(e,"boundedheight",u[3]),(!a||g[0]&32)&&b(e,"flex",u[5]),(!a||g[0]&128)&&b(e,"disable_click",u[7])},i(u){a||(E(c,u),a=!0)},o(u){W(c,u),a=!1},d(u){u&&L(e),c&&c.d(u),t[34](null),m=!1,Ke(_)}}}function xe(t){let e,l,n=!t[9]&&le(t);return{c(){n&&n.c(),e=oe()},m(r,o){n&&n.m(r,o),O(r,e,o),l=!0},p(r,o){r[9]?n&&(fe(),W(n,1,1,()=>{n=null}),re()):n?(n.p(r,o),o[0]&512&&E(n,1)):(n=le(r),n.c(),E(n,1),n.m(e.parentNode,e))},i(r){l||(E(n),l=!0)},o(r){W(n),l=!1},d(r){r&&L(e),n&&n.d(r)}}}function $e(t){let e,l,n,r,o;const d=t[26].default,f=se(d,t,t[25],null);return{c(){e=V("button"),f&&f.c(),k(e,"tabindex",l=t[9]?-1:0),k(e,"class","svelte-j5bxrl"),b(e,"hidden",t[9]),b(e,"center",t[4]),b(e,"boundedheight",t[3]),b(e,"flex",t[5]),de(e,"height","100%")},m(s,a){O(s,e,a),f&&f.m(e,null),n=!0,r||(o=v(e,"click",t[12]),r=!0)},p(s,a){f&&f.p&&(!n||a[0]&33554432)&&_e(f,d,s,s[25],n?ue(d,s[25],a,null):ae(s[25]),null),(!n||a[0]&512&&l!==(l=s[9]?-1:0))&&k(e,"tabindex",l),(!n||a[0]&512)&&b(e,"hidden",s[9]),(!n||a[0]&16)&&b(e,"center",s[4]),(!n||a[0]&8)&&b(e,"boundedheight",s[3]),(!n||a[0]&32)&&b(e,"flex",s[5])},i(s){n||(E(f,s),n=!0)},o(s){W(f,s),n=!1},d(s){s&&L(e),f&&f.d(s),r=!1,o()}}}function le(t){let e,l;return e=new Je({props:{root:t[8],upload_id:t[14],files:t[15],stream_handler:t[11]}}),{c(){Re(e.$$.fragment)},m(n,r){He(e,n,r),l=!0},p(n,r){const o={};r[0]&256&&(o.root=n[8]),r[0]&16384&&(o.upload_id=n[14]),r[0]&32768&&(o.files=n[15]),r[0]&2048&&(o.stream_handler=n[11]),e.$set(o)},i(n){l||(E(e.$$.fragment,n),l=!0)},o(n){W(e.$$.fragment,n),l=!1},d(n){Be(e,n)}}}function et(t){let e,l,n,r;const o=[$e,xe,Ze],d=[];function f(s,a){return s[0]==="clipboard"?0:s[1]&&s[10]?1:2}return e=f(t),l=d[e]=o[e](t),{c(){l.c(),n=oe()},m(s,a){d[e].m(s,a),O(s,n,a),r=!0},p(s,a){let m=e;e=f(s),e===m?d[e].p(s,a):(fe(),W(d[m],1,1,()=>{d[m]=null}),re(),l=d[e],l?l.p(s,a):(l=d[e]=o[e](s),l.c()),E(l,1),l.m(n.parentNode,n))},i(s){r||(E(l),r=!0)},o(s){W(l),r=!1},d(s){s&&L(n),d[e].d(s)}}}function tt(t,e,l){if(!t||t==="*"||t==="file/*"||Array.isArray(t)&&t.some(r=>r==="*"||r==="file/*"))return!0;let n;if(typeof t=="string")n=t.split(",").map(r=>r.trim());else if(Array.isArray(t))n=t;else return!1;return n.includes(e)||n.some(r=>{const[o]=r.split("/").map(d=>d.trim());return r.endsWith("/*")&&l.startsWith(o+"/")})}function lt(t,e,l){let{$$slots:n={},$$scope:r}=e,{filetype:o=null}=e,{dragging:d=!1}=e,{boundedheight:f=!0}=e,{center:s=!0}=e,{flex:a=!0}=e,{file_count:m="single"}=e,{disable_click:_=!1}=e,{root:h}=e,{hidden:c=!1}=e,{format:u="file"}=e,{uploading:g=!1}=e,{hidden_upload:z=null}=e,{show_progress:X=!0}=e,{max_file_size:R=null}=e,{upload:B}=e,{stream_handler:Y}=e,G,H,C;const q=Xe(),ce=["image","video","audio","text","file"],Z=i=>i.startsWith(".")||i.endsWith("/*")?i:ce.includes(i)?i+"/*":"."+i;function he(){l(20,d=!d)}function ge(){navigator.clipboard.read().then(async i=>{for(let p=0;p<i.length;p++){const A=i[p].types.find(F=>F.startsWith("image/"));if(A){i[p].getType(A).then(async F=>{const Pe=new File([F],`clipboard.${A.replace("image/","")}`);await I([Pe])});break}}})}function me(){_||z&&(l(2,z.value="",z),z.click())}async function pe(i){await Ye(),l(14,G=Math.random().toString(36).substring(2,15)),l(1,g=!0);try{const p=await B(i,h,G,R??1/0);return q("load",m==="single"?p?.[0]:p),l(1,g=!1),p||[]}catch(p){return q("error",p.message),l(1,g=!1),[]}}async function I(i){if(!i.length)return;let p=i.map(A=>new File([A],A instanceof File?A.name:"file",{type:A.type}));return l(15,H=await Te(p)),await pe(H)}async function be(i){const p=i.target;if(p.files)if(u!="blob")await I(Array.from(p.files));else{if(m==="single"){q("load",p.files[0]);return}q("load",p.files)}}async function we(i){if(l(20,d=!1),!i.dataTransfer?.files)return;const p=Array.from(i.dataTransfer.files).filter(A=>{const F="."+A.name.split(".").pop();return F&&tt(C,F,A.type)||(F&&Array.isArray(o)?o.includes(F):F===o)?!0:(q("error",`Invalid file type only ${o} allowed.`),!1)});await I(p)}function ke(i){D.call(this,t,i)}function ye(i){D.call(this,t,i)}function ve(i){D.call(this,t,i)}function ze(i){D.call(this,t,i)}function Ae(i){D.call(this,t,i)}function Fe(i){D.call(this,t,i)}function Ee(i){D.call(this,t,i)}function De(i){Oe[i?"unshift":"push"](()=>{z=i,l(2,z)})}return t.$$set=i=>{"filetype"in i&&l(0,o=i.filetype),"dragging"in i&&l(20,d=i.dragging),"boundedheight"in i&&l(3,f=i.boundedheight),"center"in i&&l(4,s=i.center),"flex"in i&&l(5,a=i.flex),"file_count"in i&&l(6,m=i.file_count),"disable_click"in i&&l(7,_=i.disable_click),"root"in i&&l(8,h=i.root),"hidden"in i&&l(9,c=i.hidden),"format"in i&&l(21,u=i.format),"uploading"in i&&l(1,g=i.uploading),"hidden_upload"in i&&l(2,z=i.hidden_upload),"show_progress"in i&&l(10,X=i.show_progress),"max_file_size"in i&&l(22,R=i.max_file_size),"upload"in i&&l(23,B=i.upload),"stream_handler"in i&&l(11,Y=i.stream_handler),"$$scope"in i&&l(25,r=i.$$scope)},t.$$.update=()=>{t.$$.dirty[0]&1&&(o==null?l(16,C=null):typeof o=="string"?l(16,C=Z(o)):(l(0,o=o.map(Z)),l(16,C=o.join(", "))))},[o,g,z,f,s,a,m,_,h,c,X,Y,ge,me,G,H,C,he,be,we,d,u,R,B,I,r,n,ke,ye,ve,ze,Ae,Fe,Ee,De]}class st extends Le{constructor(e){super(),Ge(this,e,lt,et,Qe,{filetype:0,dragging:20,boundedheight:3,center:4,flex:5,file_count:6,disable_click:7,root:8,hidden:9,format:21,uploading:1,hidden_upload:2,show_progress:10,max_file_size:22,upload:23,stream_handler:11,paste_clipboard:12,open_file_upload:13,load_files:24},null,[-1,-1])}get filetype(){return this.$$.ctx[0]}set filetype(e){this.$$set({filetype:e}),w()}get dragging(){return this.$$.ctx[20]}set dragging(e){this.$$set({dragging:e}),w()}get boundedheight(){return this.$$.ctx[3]}set boundedheight(e){this.$$set({boundedheight:e}),w()}get center(){return this.$$.ctx[4]}set center(e){this.$$set({center:e}),w()}get flex(){return this.$$.ctx[5]}set flex(e){this.$$set({flex:e}),w()}get file_count(){return this.$$.ctx[6]}set file_count(e){this.$$set({file_count:e}),w()}get disable_click(){return this.$$.ctx[7]}set disable_click(e){this.$$set({disable_click:e}),w()}get root(){return this.$$.ctx[8]}set root(e){this.$$set({root:e}),w()}get hidden(){return this.$$.ctx[9]}set hidden(e){this.$$set({hidden:e}),w()}get format(){return this.$$.ctx[21]}set format(e){this.$$set({format:e}),w()}get uploading(){return this.$$.ctx[1]}set uploading(e){this.$$set({uploading:e}),w()}get hidden_upload(){return this.$$.ctx[2]}set hidden_upload(e){this.$$set({hidden_upload:e}),w()}get show_progress(){return this.$$.ctx[10]}set show_progress(e){this.$$set({show_progress:e}),w()}get max_file_size(){return this.$$.ctx[22]}set max_file_size(e){this.$$set({max_file_size:e}),w()}get upload(){return this.$$.ctx[23]}set upload(e){this.$$set({upload:e}),w()}get stream_handler(){return this.$$.ctx[11]}set stream_handler(e){this.$$set({stream_handler:e}),w()}get paste_clipboard(){return this.$$.ctx[12]}get open_file_upload(){return this.$$.ctx[13]}get load_files(){return this.$$.ctx[24]}}export{st as U};
//# sourceMappingURL=ModifyUpload.svelte_svelte_type_style_lang-EVTRkPJ7.js.map