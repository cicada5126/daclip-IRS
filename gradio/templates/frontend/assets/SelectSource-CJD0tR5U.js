import"./Index-DvJ399W-.js";import{U as L,I as O}from"./Upload-Cp8Go_XF.js";const{SvelteComponent:Q,append:U,attr:_,detach:T,init:X,insert:Y,noop:A,safe_not_equal:Z,svg_element:S}=window.__gradio__svelte__internal;function y(c){let e,l,t,i,s;return{c(){e=S("svg"),l=S("path"),t=S("path"),i=S("line"),s=S("line"),_(l,"d","M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"),_(t,"d","M19 10v2a7 7 0 0 1-14 0v-2"),_(i,"x1","12"),_(i,"y1","19"),_(i,"x2","12"),_(i,"y2","23"),_(s,"x1","8"),_(s,"y1","23"),_(s,"x2","16"),_(s,"y2","23"),_(e,"xmlns","http://www.w3.org/2000/svg"),_(e,"width","100%"),_(e,"height","100%"),_(e,"viewBox","0 0 24 24"),_(e,"fill","none"),_(e,"stroke","currentColor"),_(e,"stroke-width","2"),_(e,"stroke-linecap","round"),_(e,"stroke-linejoin","round"),_(e,"class","feather feather-mic")},m(n,r){Y(n,e,r),U(e,l),U(e,t),U(e,i),U(e,s)},p:A,i:A,o:A,d(n){n&&T(e)}}}class x extends Q{constructor(e){super(),X(this,e,null,y,Z,{})}}const{SvelteComponent:ee,append:E,attr:w,detach:te,init:le,insert:ne,noop:N,safe_not_equal:ie,svg_element:R}=window.__gradio__svelte__internal;function se(c){let e,l,t;return{c(){e=R("svg"),l=R("path"),t=R("path"),w(l,"fill","currentColor"),w(l,"d","M12 2c-4.963 0-9 4.038-9 9c0 3.328 1.82 6.232 4.513 7.79l-2.067 1.378A1 1 0 0 0 6 22h12a1 1 0 0 0 .555-1.832l-2.067-1.378C19.18 17.232 21 14.328 21 11c0-4.962-4.037-9-9-9zm0 16c-3.859 0-7-3.141-7-7c0-3.86 3.141-7 7-7s7 3.14 7 7c0 3.859-3.141 7-7 7z"),w(t,"fill","currentColor"),w(t,"d","M12 6c-2.757 0-5 2.243-5 5s2.243 5 5 5s5-2.243 5-5s-2.243-5-5-5zm0 8c-1.654 0-3-1.346-3-3s1.346-3 3-3s3 1.346 3 3s-1.346 3-3 3z"),w(e,"xmlns","http://www.w3.org/2000/svg"),w(e,"width","100%"),w(e,"height","100%"),w(e,"viewBox","0 0 24 24")},m(i,s){ne(i,e,s),E(e,l),E(e,t)},p:N,i:N,o:N,d(i){i&&te(e)}}}class ce extends ee{constructor(e){super(),le(this,e,null,se,ie,{})}}const{SvelteComponent:oe,append:V,attr:b,check_outros:q,create_component:I,destroy_component:P,detach:k,element:M,empty:re,flush:B,group_outros:z,init:ae,insert:C,listen:W,mount_component:j,safe_not_equal:ue,space:D,toggle_class:$,transition_in:d,transition_out:h}=window.__gradio__svelte__internal;function F(c){let e,l=c[1].includes("upload"),t,i=c[1].includes("microphone"),s,n=c[1].includes("webcam"),r,g=c[1].includes("clipboard"),v,a=l&&G(c),u=i&&H(c),f=n&&J(c),o=g&&K(c);return{c(){e=M("span"),a&&a.c(),t=D(),u&&u.c(),s=D(),f&&f.c(),r=D(),o&&o.c(),b(e,"class","source-selection svelte-1ebruwp"),b(e,"data-testid","source-select")},m(m,p){C(m,e,p),a&&a.m(e,null),V(e,t),u&&u.m(e,null),V(e,s),f&&f.m(e,null),V(e,r),o&&o.m(e,null),v=!0},p(m,p){p&2&&(l=m[1].includes("upload")),l?a?(a.p(m,p),p&2&&d(a,1)):(a=G(m),a.c(),d(a,1),a.m(e,t)):a&&(z(),h(a,1,1,()=>{a=null}),q()),p&2&&(i=m[1].includes("microphone")),i?u?(u.p(m,p),p&2&&d(u,1)):(u=H(m),u.c(),d(u,1),u.m(e,s)):u&&(z(),h(u,1,1,()=>{u=null}),q()),p&2&&(n=m[1].includes("webcam")),n?f?(f.p(m,p),p&2&&d(f,1)):(f=J(m),f.c(),d(f,1),f.m(e,r)):f&&(z(),h(f,1,1,()=>{f=null}),q()),p&2&&(g=m[1].includes("clipboard")),g?o?(o.p(m,p),p&2&&d(o,1)):(o=K(m),o.c(),d(o,1),o.m(e,null)):o&&(z(),h(o,1,1,()=>{o=null}),q())},i(m){v||(d(a),d(u),d(f),d(o),v=!0)},o(m){h(a),h(u),h(f),h(o),v=!1},d(m){m&&k(e),a&&a.d(),u&&u.d(),f&&f.d(),o&&o.d()}}}function G(c){let e,l,t,i,s;return l=new L({}),{c(){e=M("button"),I(l.$$.fragment),b(e,"class","icon svelte-1ebruwp"),b(e,"aria-label","Upload file"),$(e,"selected",c[0]==="upload"||!c[0])},m(n,r){C(n,e,r),j(l,e,null),t=!0,i||(s=W(e,"click",c[6]),i=!0)},p(n,r){(!t||r&1)&&$(e,"selected",n[0]==="upload"||!n[0])},i(n){t||(d(l.$$.fragment,n),t=!0)},o(n){h(l.$$.fragment,n),t=!1},d(n){n&&k(e),P(l),i=!1,s()}}}function H(c){let e,l,t,i,s;return l=new x({}),{c(){e=M("button"),I(l.$$.fragment),b(e,"class","icon svelte-1ebruwp"),b(e,"aria-label","Record audio"),$(e,"selected",c[0]==="microphone")},m(n,r){C(n,e,r),j(l,e,null),t=!0,i||(s=W(e,"click",c[7]),i=!0)},p(n,r){(!t||r&1)&&$(e,"selected",n[0]==="microphone")},i(n){t||(d(l.$$.fragment,n),t=!0)},o(n){h(l.$$.fragment,n),t=!1},d(n){n&&k(e),P(l),i=!1,s()}}}function J(c){let e,l,t,i,s;return l=new ce({}),{c(){e=M("button"),I(l.$$.fragment),b(e,"class","icon svelte-1ebruwp"),b(e,"aria-label","Capture from camera"),$(e,"selected",c[0]==="webcam")},m(n,r){C(n,e,r),j(l,e,null),t=!0,i||(s=W(e,"click",c[8]),i=!0)},p(n,r){(!t||r&1)&&$(e,"selected",n[0]==="webcam")},i(n){t||(d(l.$$.fragment,n),t=!0)},o(n){h(l.$$.fragment,n),t=!1},d(n){n&&k(e),P(l),i=!1,s()}}}function K(c){let e,l,t,i,s;return l=new O({}),{c(){e=M("button"),I(l.$$.fragment),b(e,"class","icon svelte-1ebruwp"),b(e,"aria-label","Paste from clipboard"),$(e,"selected",c[0]==="clipboard")},m(n,r){C(n,e,r),j(l,e,null),t=!0,i||(s=W(e,"click",c[9]),i=!0)},p(n,r){(!t||r&1)&&$(e,"selected",n[0]==="clipboard")},i(n){t||(d(l.$$.fragment,n),t=!0)},o(n){h(l.$$.fragment,n),t=!1},d(n){n&&k(e),P(l),i=!1,s()}}}function fe(c){let e,l,t=c[2].length>1&&F(c);return{c(){t&&t.c(),e=re()},m(i,s){t&&t.m(i,s),C(i,e,s),l=!0},p(i,[s]){i[2].length>1?t?(t.p(i,s),s&4&&d(t,1)):(t=F(i),t.c(),d(t,1),t.m(e.parentNode,e)):t&&(z(),h(t,1,1,()=>{t=null}),q())},i(i){l||(d(t),l=!0)},o(i){h(t),l=!1},d(i){i&&k(e),t&&t.d(i)}}}function _e(c,e,l){let t,{sources:i}=e,{active_source:s}=e,{handle_clear:n=()=>{}}=e,{handle_select:r=()=>{}}=e;async function g(o){n(),l(0,s=o),r(o)}const v=()=>g("upload"),a=()=>g("microphone"),u=()=>g("webcam"),f=()=>g("clipboard");return c.$$set=o=>{"sources"in o&&l(1,i=o.sources),"active_source"in o&&l(0,s=o.active_source),"handle_clear"in o&&l(4,n=o.handle_clear),"handle_select"in o&&l(5,r=o.handle_select)},c.$$.update=()=>{c.$$.dirty&2&&l(2,t=[...new Set(i)])},[s,i,t,g,n,r,v,a,u,f]}class pe extends oe{constructor(e){super(),ae(this,e,_e,fe,ue,{sources:1,active_source:0,handle_clear:4,handle_select:5})}get sources(){return this.$$.ctx[1]}set sources(e){this.$$set({sources:e}),B()}get active_source(){return this.$$.ctx[0]}set active_source(e){this.$$set({active_source:e}),B()}get handle_clear(){return this.$$.ctx[4]}set handle_clear(e){this.$$set({handle_clear:e}),B()}get handle_select(){return this.$$.ctx[5]}set handle_select(e){this.$$set({handle_select:e}),B()}}export{pe as S,ce as W};
//# sourceMappingURL=SelectSource-CJD0tR5U.js.map