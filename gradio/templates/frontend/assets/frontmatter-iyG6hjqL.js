import{s as m,t as a,f as i,c as s,p,S as l}from"./Index-DXAmymL2.js";import{yaml as f}from"./yaml-DsCXHVTH.js";import"./index-D6iiusuW.js";import"./svelte/svelte.js";import"./Button-BM3Gfoxv.js";import"./Index-DvJ399W-.js";import"./_commonjsHelpers-BosuxZz1.js";import"./Check-Ck0iADAu.js";import"./Copy-ZPOKSMtK.js";import"./DownloadLink-C3h3PR_b.js";import"./file-url-DulxDZ3L.js";import"./BlockLabel-S59nE2jv.js";import"./Empty-CgtesKU7.js";import"./Example-Wp-_4AVX.js";const n=/^---\s*$/m,L={defineNodes:[{name:"Frontmatter",block:!0},"FrontmatterMark"],props:[m({Frontmatter:[a.documentMeta,a.monospace],FrontmatterMark:a.processingInstruction}),i.add({Frontmatter:s,FrontmatterMark:()=>null})],wrap:p(t=>{const{parser:e}=l.define(f);return t.type.name==="Frontmatter"?{parser:e,overlay:[{from:t.from+4,to:t.to-4}]}:null}),parseBlock:[{name:"Frontmatter",before:"HorizontalRule",parse:(t,e)=>{let r;const o=new Array;if(t.lineStart===0&&n.test(e.text)){for(o.push(t.elt("FrontmatterMark",0,4));t.nextLine();)if(n.test(e.text)){r=t.lineStart+4;break}return r!==void 0&&(o.push(t.elt("FrontmatterMark",r-4,r)),t.addElement(t.elt("Frontmatter",0,r,o))),!0}return!1}}]};export{L as frontmatter};
//# sourceMappingURL=frontmatter-iyG6hjqL.js.map
