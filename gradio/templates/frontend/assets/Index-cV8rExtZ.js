import{M as q}from"./Index.svelte_svelte_type_style_lang-Cp94WWVp.js";import{S as E}from"./Index-DvJ399W-.js";import{B as I}from"./Button-BM3Gfoxv.js";import{default as p}from"./Example-DZJukuNR.js";import{M as te}from"./Example.svelte_svelte_type_style_lang-DyHaW-Bg.js";import"./Blocks-BrGSw8d-.js";import"./index-D6iiusuW.js";import"./svelte/svelte.js";import"./_commonjsHelpers-BosuxZz1.js";import"./prism-python-b7Hj1w62.js";const{SvelteComponent:A,assign:D,attr:F,create_component:c,destroy_component:d,detach:M,element:G,flush:u,get_spread_object:H,get_spread_update:J,init:K,insert:B,mount_component:k,safe_not_equal:L,space:N,toggle_class:S,transition_in:b,transition_out:v}=window.__gradio__svelte__internal;function O(s){let e,a,i,r,_;const f=[{autoscroll:s[8].autoscroll},{i18n:s[8].i18n},s[4],{variant:"center"}];let h={};for(let t=0;t<f.length;t+=1)h=D(h,f[t]);return e=new E({props:h}),e.$on("clear_status",s[12]),r=new q({props:{min_height:s[4]&&s[4].status!=="complete",value:s[3],elem_classes:s[1],visible:s[2],rtl:s[5],latex_delimiters:s[9],sanitize_html:s[6],line_breaks:s[7],header_links:s[10]}}),r.$on("change",s[13]),{c(){c(e.$$.fragment),a=N(),i=G("div"),c(r.$$.fragment),F(i,"class","svelte-1ed2p3z"),S(i,"pending",s[4]?.status==="pending")},m(t,n){k(e,t,n),B(t,a,n),B(t,i,n),k(r,i,null),_=!0},p(t,n){const g=n&272?J(f,[n&256&&{autoscroll:t[8].autoscroll},n&256&&{i18n:t[8].i18n},n&16&&H(t[4]),f[3]]):{};e.$set(g);const m={};n&16&&(m.min_height=t[4]&&t[4].status!=="complete"),n&8&&(m.value=t[3]),n&2&&(m.elem_classes=t[1]),n&4&&(m.visible=t[2]),n&32&&(m.rtl=t[5]),n&512&&(m.latex_delimiters=t[9]),n&64&&(m.sanitize_html=t[6]),n&128&&(m.line_breaks=t[7]),n&1024&&(m.header_links=t[10]),r.$set(m),(!_||n&16)&&S(i,"pending",t[4]?.status==="pending")},i(t){_||(b(e.$$.fragment,t),b(r.$$.fragment,t),_=!0)},o(t){v(e.$$.fragment,t),v(r.$$.fragment,t),_=!1},d(t){t&&(M(a),M(i)),d(e,t),d(r)}}}function P(s){let e,a;return e=new I({props:{visible:s[2],elem_id:s[0],elem_classes:s[1],container:!1,allow_overflow:!0,$$slots:{default:[O]},$$scope:{ctx:s}}}),{c(){c(e.$$.fragment)},m(i,r){k(e,i,r),a=!0},p(i,[r]){const _={};r&4&&(_.visible=i[2]),r&1&&(_.elem_id=i[0]),r&2&&(_.elem_classes=i[1]),r&18430&&(_.$$scope={dirty:r,ctx:i}),e.$set(_)},i(i){a||(b(e.$$.fragment,i),a=!0)},o(i){v(e.$$.fragment,i),a=!1},d(i){d(e,i)}}}function Q(s,e,a){let{label:i}=e,{elem_id:r=""}=e,{elem_classes:_=[]}=e,{visible:f=!0}=e,{value:h=""}=e,{loading_status:t}=e,{rtl:n=!1}=e,{sanitize_html:g=!0}=e,{line_breaks:m=!1}=e,{gradio:o}=e,{latex_delimiters:w}=e,{header_links:z=!1}=e;const C=()=>o.dispatch("clear_status",t),j=()=>o.dispatch("change");return s.$$set=l=>{"label"in l&&a(11,i=l.label),"elem_id"in l&&a(0,r=l.elem_id),"elem_classes"in l&&a(1,_=l.elem_classes),"visible"in l&&a(2,f=l.visible),"value"in l&&a(3,h=l.value),"loading_status"in l&&a(4,t=l.loading_status),"rtl"in l&&a(5,n=l.rtl),"sanitize_html"in l&&a(6,g=l.sanitize_html),"line_breaks"in l&&a(7,m=l.line_breaks),"gradio"in l&&a(8,o=l.gradio),"latex_delimiters"in l&&a(9,w=l.latex_delimiters),"header_links"in l&&a(10,z=l.header_links)},s.$$.update=()=>{s.$$.dirty&2304&&o.dispatch("change")},[r,_,f,h,t,n,g,m,o,w,z,i,C,j]}class x extends A{constructor(e){super(),K(this,e,Q,P,L,{label:11,elem_id:0,elem_classes:1,visible:2,value:3,loading_status:4,rtl:5,sanitize_html:6,line_breaks:7,gradio:8,latex_delimiters:9,header_links:10})}get label(){return this.$$.ctx[11]}set label(e){this.$$set({label:e}),u()}get elem_id(){return this.$$.ctx[0]}set elem_id(e){this.$$set({elem_id:e}),u()}get elem_classes(){return this.$$.ctx[1]}set elem_classes(e){this.$$set({elem_classes:e}),u()}get visible(){return this.$$.ctx[2]}set visible(e){this.$$set({visible:e}),u()}get value(){return this.$$.ctx[3]}set value(e){this.$$set({value:e}),u()}get loading_status(){return this.$$.ctx[4]}set loading_status(e){this.$$set({loading_status:e}),u()}get rtl(){return this.$$.ctx[5]}set rtl(e){this.$$set({rtl:e}),u()}get sanitize_html(){return this.$$.ctx[6]}set sanitize_html(e){this.$$set({sanitize_html:e}),u()}get line_breaks(){return this.$$.ctx[7]}set line_breaks(e){this.$$set({line_breaks:e}),u()}get gradio(){return this.$$.ctx[8]}set gradio(e){this.$$set({gradio:e}),u()}get latex_delimiters(){return this.$$.ctx[9]}set latex_delimiters(e){this.$$set({latex_delimiters:e}),u()}get header_links(){return this.$$.ctx[10]}set header_links(e){this.$$set({header_links:e}),u()}}export{p as BaseExample,q as BaseMarkdown,te as MarkdownCode,x as default};
//# sourceMappingURL=Index-cV8rExtZ.js.map
