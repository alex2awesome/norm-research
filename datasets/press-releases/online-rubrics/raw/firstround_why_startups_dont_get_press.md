<!-- URL: https://review.firstround.com/why-most-startups-dont-get-press/ -->
<!-- HTTP: 200 SIZE:   143239 -->
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">

    <title>Why Most Startups Don&#x27;t &#x27;Get&#x27; Press</title>

    <meta name="HandheldFriendly" content="True" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />

    <meta name="theme-color" content="#62DCAD">

    <style>

@font-face {
  font-family: 'Leif Bold';
  src: url('/assets/src/fonts/leif-bold.woff2') format('woff2');
  font-weight:  600;
  font-stretch: normal;
  font-display: swap;
}
@font-face {
  font-family: 'Leif Regular';
  src: url('/assets/src/fonts/leif-regular.woff2') format('woff2');
  font-weight:  400;
  font-stretch: normal;
  font-display: swap;
}
@font-face {
  font-family: 'Leif Italic';
  src: url('/assets/src/fonts/leif-italic.woff2') format('woff2');
  font-weight:  400;
  font-style:   italic;
  font-stretch: normal;
  font-display: swap;
}

@font-face {
  font-family: 'Skandia Bold';
  src: url('/assets/src/fonts/skandia-bold.woff2') format('woff2');
  font-weight:  600;
  font-style:   normal;
  font-stretch: normal;
  font-display: swap;
}

@font-face {
  font-family: 'Skandia Regular';
  src: url('/assets/src/fonts/skandia-regular.woff2') format('woff2');
  font-weight:  400;
  font-stretch: normal;
  font-display: swap;
}

@font-face {
  font-family: 'Skandia Book';
  src: url('/assets/src/fonts/skandia-book.woff2') format('woff2');
  font-stretch: normal;
  font-display: swap;
}
@font-face {
  font-family: 'Skandia Italic';
  src: url('/assets/src/fonts/skandia-italic.woff2') format('woff2');
  font-weight:  400;
  font-style:   italic;
  font-stretch: normal;
  font-display: swap;
}

@font-face {
  font-family: 'Skandia Medium';
  src: url('/assets/src/fonts/skandia-medium.woff2') format('woff2');
  font-stretch: normal;
  font-display: swap;
}

</style>
    <script defer src="/assets/built/app.js?v=1f025fdd46"></script>

    <link rel="stylesheet" type="text/css" href="/assets/built/app.css?v=1f025fdd46" />

    <script>
  // Global values needed
  const themeGlobal = {
    currentPage: parseInt(''),
    nextPage: parseInt(''),
    nextPageLink: '',
    maxPages: parseInt(''), 
    lastPage: `` === `` ? true : false,
    postsPerPage: parseInt('12'),
    scrollPos: 0,
  }

  // Calculate contrast & HSL value;
  function getColorInfo(hexcolor) {
    // get contrast
    if (hexcolor.slice(0, 1) === '#') { hexcolor = hexcolor.slice(1); }
    if (hexcolor.length === 3) { hexcolor = hexcolor.split('').map(function (hex) { return hex + hex;}).join(''); }
    let r = parseInt(hexcolor.substr(0,2),16), g = parseInt(hexcolor.substr(2,2),16), b = parseInt(hexcolor.substr(4,2),16);
    let yiq = ((r * 299) + (g * 587) + (b * 114)) / 1000;
    const colorContrast = (yiq >= 128) ? '#000' : '#fff';

    //get HSL
    r /= 255, g /= 255, b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    let h, s, l = (max + min)  /  2;  
    if ( max == min ) { h = s = 0; } else {
      let d = max - min;
      s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
      switch(max){
        case r: h = (g - b) / d + (g < b ? 6 : 0); break;
        case g: h = (b - r) / d + 2; break;
        case b: h = (r - g) / d + 4; break;
      }
      h /= 6;
    }
    const colorHSL = [Math.round(h * 360), Math.round(s * 100), Math.round(l * 100)];

    // return
    return { colorContrast, colorHSL }
  };

  const brandColor = getColorInfo("#62DCAD");
  let style = document.createElement('style');
  style.innerHTML = `:root { 
    --color-brand-contrast: ${brandColor.colorContrast}; 
    --color-brand-h: ${brandColor.colorHSL[0]};
    --color-brand-s: ${brandColor.colorHSL[1]}%;
    --color-brand-l: ${brandColor.colorHSL[2]}%;
    --color-brand-hsl: ${brandColor.colorHSL[0]} ${brandColor.colorHSL[1]}% ${brandColor.colorHSL[2]}%;
  }`
  document.getElementsByTagName('head')[0].appendChild(style);
</script>


<style>body { overflow-y: scroll!important; }</style>
<style>.post-template .outpost-product-box { margin-top: 1.33em; }</style>
<style>.fluid-width-podcast-wrapper + .outpost-pub-container { margin-top: 2rem; }</style>
    <meta name="description" content="PR guru Brooke Hammerling gives step-by-step advice for startups looking to get good press coverage without an agency.">
    <link rel="icon" href="https://review.firstround.com/content/images/size/w256h256/format/png/2024/03/First-Round-icon-blk.svg" type="image/png">
    <link rel="canonical" href="https://review.firstround.com/why-most-startups-dont-get-press/">
    <meta name="referrer" content="no-referrer-when-downgrade">
    
    <meta property="og:site_name" content="First Round">
    <meta property="og:type" content="article">
    <meta property="og:title" content="Why Most Startups Don&#x27;t &#x27;Get&#x27; Press">
    <meta property="og:description" content="PR guru Brooke Hammerling gives step-by-step advice for startups looking to get good press coverage without an agency.">
    <meta property="og:url" content="https://review.firstround.com/why-most-startups-dont-get-press/">
    <meta property="og:image" content="https://review.firstround.com/content/images/size/w1200/2056/firstround-2fty8bg9jtsoqhsupmu17l_54832618_thumbnail.jpg">
    <meta property="article:published_time" content="2015-02-05T22:15:14.000Z">
    <meta property="article:modified_time" content="2024-02-29T17:30:33.000Z">
    <meta property="article:tag" content="PR &amp;amp; Marketing">
    <meta property="article:tag" content="Women">
    <meta property="article:tag" content="Product marketing">
    <meta property="article:tag" content="Storytelling">
    <meta property="article:tag" content="Brand">
    <meta property="article:tag" content="PR">
    <meta property="article:tag" content="Brooke Hammerling">
    <meta property="article:tag" content="Brew Media Relations">
    
    <meta property="article:publisher" content="https://www.facebook.com/firstroundcapital/">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="Why Most Startups Don&#x27;t &#x27;Get&#x27; Press">
    <meta name="twitter:description" content="PR guru Brooke Hammerling gives step-by-step advice for startups looking to get good press coverage without an agency.">
    <meta name="twitter:url" content="https://review.firstround.com/why-most-startups-dont-get-press/">
    <meta name="twitter:image" content="https://review.firstround.com/content/images/size/w1200/2056/firstround-2flwcokxqmrrcvhenijwy0_54832618_thumbnail.jpg">
    <meta name="twitter:label1" content="Written by">
    <meta name="twitter:data1" content="First Round Staff">
    <meta name="twitter:label2" content="Filed under">
    <meta name="twitter:data2" content="PR &amp; Marketing, Women, Product marketing, Storytelling, Brand, PR, Brooke Hammerling, Brew Media Relations">
    <meta name="twitter:site" content="@firstround">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="835">
    
    <script type="application/ld+json">
{
    "@context": "https://schema.org",
    "@type": "Article",
    "publisher": {
        "@type": "Organization",
        "name": "First Round",
        "url": "https://review.firstround.com/",
        "logo": {
            "@type": "ImageObject",
            "url": "https://review.firstround.com/content/images/2024/02/Reviewlogo.png"
        }
    },
    "author": {
        "@type": "Person",
        "name": "First Round Staff",
        "url": "https://review.firstround.com/404/",
        "sameAs": []
    },
    "headline": "Why Most Startups Don&#x27;t &#x27;Get&#x27; Press",
    "url": "https://review.firstround.com/why-most-startups-dont-get-press/",
    "datePublished": "2015-02-05T22:15:14.000Z",
    "dateModified": "2024-02-29T17:30:33.000Z",
    "image": {
        "@type": "ImageObject",
        "url": "https://review.firstround.com/content/images/size/w1200/2056/firstround/lwcokxqmrrcvhenijwy0_54832618_thumbnail.jpg",
        "width": 1200,
        "height": 835
    },
    "keywords": "PR & Marketing, Women, Product marketing, Storytelling, Brand, PR, Brooke Hammerling, Brew Media Relations",
    "description": "PR guru Brooke Hammerling gives step-by-step advice for startups looking to get good press coverage without an agency.",
    "mainEntityOfPage": "https://review.firstround.com/why-most-startups-dont-get-press/"
}
    </script>

    <meta name="generator" content="Ghost 6.19">
    <link rel="alternate" type="application/rss+xml" title="First Round" href="https://review.firstround.com/glossary/rss/">
    <script defer src="https://cdn.jsdelivr.net/ghost/portal@~2.64/umd/portal.min.js" data-i18n="true" data-ghost="https://review.firstround.com/" data-key="903b94f11f111f32d9d1bf2617" data-api="https://review.firstround.com/ghost/api/content/" data-locale="en" crossorigin="anonymous"></script><style id="gh-members-styles">.gh-post-upgrade-cta-content,
.gh-post-upgrade-cta {
    display: flex;
    flex-direction: column;
    align-items: center;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
    text-align: center;
    width: 100%;
    color: #ffffff;
    font-size: 16px;
}

.gh-post-upgrade-cta-content {
    border-radius: 8px;
    padding: 40px 4vw;
}

.gh-post-upgrade-cta h2 {
    color: #ffffff;
    font-size: 28px;
    letter-spacing: -0.2px;
    margin: 0;
    padding: 0;
}

.gh-post-upgrade-cta p {
    margin: 20px 0 0;
    padding: 0;
}

.gh-post-upgrade-cta small {
    font-size: 16px;
    letter-spacing: -0.2px;
}

.gh-post-upgrade-cta a {
    color: #ffffff;
    cursor: pointer;
    font-weight: 500;
    box-shadow: none;
    text-decoration: underline;
}

.gh-post-upgrade-cta a:hover {
    color: #ffffff;
    opacity: 0.8;
    box-shadow: none;
    text-decoration: underline;
}

.gh-post-upgrade-cta a.gh-btn {
    display: block;
    background: #ffffff;
    text-decoration: none;
    margin: 28px 0 0;
    padding: 8px 18px;
    border-radius: 4px;
    font-size: 16px;
    font-weight: 600;
}

.gh-post-upgrade-cta a.gh-btn:hover {
    opacity: 0.92;
}</style>
    <script defer src="https://cdn.jsdelivr.net/ghost/sodo-search@~1.8/umd/sodo-search.min.js" data-key="903b94f11f111f32d9d1bf2617" data-styles="https://cdn.jsdelivr.net/ghost/sodo-search@~1.8/umd/main.css" data-sodo-search="https://review.firstround.com/" data-locale="en" crossorigin="anonymous"></script>
    
    <link href="https://review.firstround.com/webmentions/receive/" rel="webmention">
    <script defer src="/public/cards.min.js?v=1f025fdd46"></script>
    <link rel="stylesheet" type="text/css" href="/public/cards.min.css?v=1f025fdd46">
    <script defer src="/public/member-attribution.min.js?v=1f025fdd46"></script><style>:root {--ghost-accent-color: #62DCAD;}</style>
    <meta name="robots" content="max-image-preview:standard"> 

<script>
(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start': new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0], j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src='https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);})(window,document,'script','dataLayer','GTM-5SRRMHZ8');
</script>

<script defer data-domain="review.firstround.com" src="https://plausible.io/js/script.hash.outbound-links.js"></script>
<script>window.plausible = window.plausible || function() { (window.plausible.q = window.plausible.q || []).push(arguments) }</script>

<script type="javascript">
(function(l,e,a,p) {
if (window.Sprig) return;
window.Sprig = function(){S._queue.push(arguments)}
var S = window.Sprig;S.appId = a;S._queue = [];window.UserLeap=S;
a=l.createElement('script');
a.async=1;a.src=e+'?id='+S.appId;
p=l.getElementsByTagName('script')[0];
p.parentNode.insertBefore(a, p);
})(document, 'https://cdn.sprig.com/shim.js', 'HWgeFdudn');
</script>
  </head>
  <body class="post-template tag-pr-and-marketing tag-women tag-product-marketing tag-storytelling tag-brand tag-pr tag-brooke-hammerling tag-brew-media-relations tag-hash-wp tag-hash-wp-post tag-hash-import-2023-12-22-15-59 relative overflow-x-hidden bg-backgr text-typogr scroll-smooth "
    x-data="{showMenu : false}" :class="{ 'overflow-hidden': showMenu, '': !showMenu}"
    data-user="visitor"
    @keydown.escape="menuOpen = false"
  >

    <script>
  document.addEventListener("DOMContentLoaded", function () {
    if (window.innerWidth >= 1024) { // Apply only on desktop
      let lastScrollTop = 0;
      const body = document.body;

      window.addEventListener("scroll", function () {
        let scrollTop = window.pageYOffset || document.documentElement.scrollTop;

        if (scrollTop === 0) {
          // Remove class when at the top
          body.classList.remove("header-sticky");
        } else if (scrollTop > lastScrollTop) {
          // Scrolling down
          body.classList.remove("header-sticky");
        } else {
          // Scrolling up
          body.classList.add("header-sticky");
        }
        lastScrollTop = scrollTop;
      });
    }
  });
</script>


<header data-label="Desktop Site Header" class="header-desktop hidden lg:block pt-5 relative pb-15 ">
  <div class="flex items-center justify-center mx-auto max-w-7xl">

  <div data-label="Branding" class="h-[76px] opacity-0">
    <a class="shadow-none cursor-pointer outline-hidden focus:outline-hidden focus:shadow-none active:outline-hidden fr-logo" href="https://review.firstround.com" aria-label="First Round">
      <svg width="38" height="76" viewBox="0 0 38 76" class="fill-current " fill="none" xmlns="http://www.w3.org/2000/svg">
<g clip-path="url(#clip0_635_302)">
<path d="M4.35052 0L0 19.8845H12.4484V76H37.4178V0H4.35052Z" fill="#000503"/>
<path d="M29.7616 61.1697V64.3195L25.3385 66.7489V67.6034H29.7616V70.3176H18.2012V66.1625C18.2012 64.8724 18.4692 63.8672 19.0221 63.1132C19.5918 62.3425 20.5133 61.8399 21.6693 61.8399C23.8641 61.8399 24.7186 63.2473 24.9866 64.1017L29.7616 61.1697ZM20.597 65.1572C20.3457 65.4923 20.2955 65.9279 20.2955 66.5479V67.6201H23.261V66.5479C23.261 65.9279 23.1939 65.4923 22.9426 65.1572C22.6913 64.8054 22.373 64.5709 21.7698 64.5709C21.1667 64.5709 20.8483 64.8054 20.597 65.1572Z" fill="white"/>
<path d="M26.2773 52.0219V57.9027C27.4669 57.7854 28.003 56.8806 28.003 55.7916C28.003 54.7026 27.5339 53.7811 27.2155 53.312L28.6061 52.2565C28.8072 52.5245 29.9465 53.6136 29.9465 56.177C29.9465 58.7404 28.3716 60.5833 25.6406 60.5833C22.9097 60.5833 21.3516 58.5728 21.3516 56.1267C21.3516 53.6806 22.9767 51.9884 25.3893 51.9884C25.7914 51.9884 26.2773 52.0219 26.2773 52.0219ZM24.7527 54.4848C23.9987 54.5015 23.094 55.0544 23.094 56.177C23.094 57.2995 23.9652 57.8524 24.7527 57.8859V54.4848Z" fill="white"/>
<path d="M21.502 41.9023L29.7618 45.3035V47.8166L21.502 51.2009V48.403L26.7293 46.5265L21.502 44.6668V41.9023Z" fill="white"/>
<path d="M20.5468 39.2217C20.5468 40.1264 19.9939 40.7464 19.1562 40.7464C18.3185 40.7464 17.7656 40.1264 17.7656 39.2217C17.7656 38.317 18.3353 37.6971 19.1562 37.6971C19.9772 37.6971 20.5468 38.317 20.5468 39.2217ZM29.7617 37.8814V40.5621H21.5018V37.8814H29.7617Z" fill="white"/>
<path d="M26.2773 27.4937V33.3744C27.4669 33.2572 28.003 32.3524 28.003 31.2634C28.003 30.1744 27.5339 29.2529 27.2155 28.7838L28.6061 27.7283C28.8072 27.9963 29.9465 29.0854 29.9465 31.6488C29.9465 34.2122 28.3716 36.0551 25.6406 36.0551C22.9097 36.0551 21.3516 34.0446 21.3516 31.5985C21.3516 29.1524 22.9767 27.4602 25.3893 27.4602C25.7914 27.4602 26.2773 27.4937 26.2773 27.4937ZM24.7527 29.9566C23.9987 29.9733 23.094 30.5262 23.094 31.6488C23.094 32.7713 23.9652 33.3242 24.7527 33.3577V29.9566Z" fill="white"/>
<path d="M21.502 13.0181L29.7618 15.7323V18.2621L24.702 19.8705L29.7618 21.4957V24.0256L21.502 26.7398V24.0256L26.7795 22.568L21.502 20.9763V18.7815L26.7795 17.1899L21.502 15.7323V13.0181Z" fill="white"/>
</g>
<defs>
<clipPath id="clip0_635_302">
<rect width="37.4178" height="76" fill="white"/>
</clipPath>
</defs>
</svg>
    </a>
  </div>

  <ul class="flex flex-row ml-[-70px]" data-nav-items>
      <li class="nav-articles px-4 flex items-center nav-item"
          data-label="Articles" data-length="8" data-slug="articles">
        <a class="text-sm px-2 leading-none h-[30px] flex items-center justify-center" href="https://review.firstround.com/articles/">
          <span>Articles</span>
        </a>
        <div id="tag-content-articles" class="dropdown-content"></div>
      </li>
      <li class="nav-curated-reads px-4 flex items-center nav-item"
          data-label="Curated Reads" data-length="13" data-slug="curated-reads">
        <a class="text-sm px-2 leading-none h-[30px] flex items-center justify-center" href="https://review.firstround.com/collections/">
          <span>Curated Reads</span>
        </a>
        <div id="tag-content-curated-reads" class="dropdown-content"></div>
      </li>
      <li class="nav-paths-to-pmf px-4 flex items-center nav-item"
          data-label="Paths to PMF" data-length="12" data-slug="paths-to-pmf">
        <a class="text-sm px-2 leading-none h-[30px] flex items-center justify-center" href="https://review.firstround.com/series/product-market-fit/">
          <span>Paths to PMF</span>
        </a>
        <div id="tag-content-paths-to-pmf" class="dropdown-content"></div>
      </li>
      <li class="nav-newsletter px-4 flex items-center nav-item"
          data-label="Newsletter" data-length="10" data-slug="newsletter">
        <a class="text-sm px-2 leading-none h-[30px] flex items-center justify-center" href="https://review.firstround.com/newsletter/">
          <span>Newsletter</span>
        </a>
        <div id="tag-content-newsletter" class="dropdown-content"></div>
      </li>
      <li class="nav-podcast px-4 flex items-center nav-item"
          data-label="Podcast" data-length="7" data-slug="podcast">
        <a class="text-sm px-2 leading-none h-[30px] flex items-center justify-center" href="https://review.firstround.com/podcast/">
          <span>Podcast</span>
        </a>
        <div id="tag-content-podcast" class="dropdown-content"></div>
      </li>
      <li class="nav-executive-function px-4 flex items-center is-subitem dropdown-subitem" 
          data-label="-Executive Function" data-length="19">
        <a class="text-sm px-2 leading-none h-[30px] flex items-center justify-center" href="https://review.firstround.com/executive-function/">
          <span>-Executive Function</span>
        </a>
      </li>
      <li class="nav-in-depth px-4 flex items-center is-subitem dropdown-subitem" 
          data-label="-In Depth" data-length="9">
        <a class="text-sm px-2 leading-none h-[30px] flex items-center justify-center" href="https://review.firstround.com/podcast/">
          <span>-In Depth</span>
        </a>
      </li>

     <li class="nav- nav-current  px-4 flex items-center max-lg:py-6 max-lg:px-0 max-lg:text-left fr-signin"
      data-label="" data-length="">
      <a href="/signin/" class="text-sm lg:px-2 leading-none h-[30px] flex items-center justify-center max-lg:font-skandia-bold max-lg:text-3xl max-lg:leading-[44.8px] max-lg:tracking-[-0.03em] max-lg:cursor-pointer max-lg:text-white" data-portal="signin"><span>Sign In</span></a>
    </li>
</ul>

<style>
  /* Dropdown parent - maintain existing layout */
  .dropdown-parent {
    position: relative;
  }

  /* Dropdown arrow indicator */
  .dropdown-parent .dropbtn {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
  }

  .dropdown-parent .dropdown-arrow {
    opacity: 0.6;
    transition: transform 0.3s ease, opacity 0.3s ease;
    margin-left: 0.25rem;
    flex-shrink: 0;
  }

  /* Dropdown content - positioned absolutely, doesn't affect layout */
  .dropdown-content {
    display: none;
    position: absolute;
    top: 100%;
    left: 0;
    background-color: var(--color-body-bg, #fff);
    min-width: 200px;
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    z-index: 1000;
    padding: 0.5rem 6px;
    margin-top: 0.5rem;
    border-radius: 0.25rem;
    border: 1px solid rgba(0, 0, 0, 0.05);
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.3s ease, visibility 0.3s ease, transform 0.3s ease;
    transform: translateY(-10px);
  }

  /* Desktop: Show dropdown on hover */
  @media (min-width: 1024px) {
    .dropdown-parent:hover .dropdown-arrow {
      opacity: 1;
      transform: rotate(180deg);
    }

    .dropdown-parent:hover .dropdown-content {
      display: block;
      opacity: 1;
      visibility: visible;
      transform: translateY(0);
      top: 48px;
    }
  }

  .dropdown-content.show {
    display: block;
    opacity: 1;
    visibility: visible;
    transform: translateY(0);
  }

  /* Dropdown items - match theme styling */
  .dropdown-content .dropdown-item {
    display: block;
    transition: opacity 0.3s ease;
    padding: 0 6px;
  }

  .dropdown-content .dropdown-item a {
    display: block;
    padding: 0.625rem 0.625rem;
    text-decoration: none;
    color: var(--color-typography, #000);
    font-family: var(--font-body);
    font-size: 0.875rem;
    transition: all 0.3s ease;
  }

  .dropdown-content .dropdown-item a:hover {
    background-color: rgba(0, 0, 0, 0.03);
    font-weight: 500;
  }

  /* Mobile: dropdown items should look like regular menu items */
  @media (max-width: 1023px) {
    .dropdown-content .dropdown-item {
      padding: 0 !important;
      background-color: transparent !important;
    }

    .dropdown-content .dropdown-item a {
      background-color: transparent !important;
      padding: 0.5rem 0 !important;
    }

    .dropdown-content .dropdown-item a:hover {
      background-color: transparent !important;
    }
  }

  /* Mobile: Disable hover completely, only allow click on arrow */
  @media (max-width: 1023px) {
    /* Disable ALL hover effects on mobile - prevent any hover state */
    .dropdown-parent:hover .dropdown-content.show {
      display: block !important;
      opacity: 1 !important;
      visibility: visible !important;
      transform: none !important;
      transition: none !important;
      pointer-events: auto !important;
    }

    .dropdown-parent:hover .dropdown-arrow {
      transform: rotate(-90deg) !important;
      opacity: 0.8 !important;
      transition: none !important;
    }
  }

  /* Touch devices: Completely disable hover (more specific than max-width) */
  @media (hover: none) and (pointer: coarse) {
    .dropdown-parent:hover .dropdown-content {
      display: none !important;
      opacity: 0 !important;
      visibility: hidden !important;
      transform: none !important;
      transition: none !important;
      pointer-events: none !important;
    }

    .dropdown-parent:hover .dropdown-arrow {
      transform: rotate(-90deg) !important;
      opacity: 0.8 !important;
      transition: none !important;
    }
  }

  @media (max-width: 1023px) {
    .dropdown-content {
      position: static !important;
      background-color: transparent !important;
      box-shadow: none !important;
      border: none !important;
      margin-top: 0 !important;
      padding-left: 1rem;
      padding-right: 0 !important;
      transform: none !important;
      transition: none !important;
      display: none !important;
      opacity: 1 !important;
      visibility: hidden !important;
      min-width: auto !important;
      pointer-events: none !important;
    }

    /* Only show when .show class is added via JS click */
    .dropdown-content.show {
      display: block !important;
      opacity: 1 !important;
      visibility: visible !important;
      pointer-events: auto !important;
    }

    .dropdown-parent .dropdown-arrow {
      transform: rotate(-90deg);
      cursor: pointer;
      pointer-events: auto !important;
      opacity: 0.8;
      min-width: 32px;
      min-height: 32px;
      padding: 8px;
      margin-left: 0.5rem;
      display: inline-flex !important;
      align-items: center;
      justify-content: center;
      -webkit-tap-highlight-color: transparent;
      touch-action: manipulation;
    }

    .dropdown-parent .dropdown-arrow:active {
      opacity: 1;
    }

    .dropdown-parent .dropdown-arrow.show {
      transform: rotate(90deg);
      opacity: 1;
    }

    /* Mobile menu specific styling */
    [data-label="Site Menu"] .dropdown-parent .dropbtn {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
    }

    [data-label="Site Menu"] .dropdown-content {
      position: static !important;
      background-color: transparent !important;
      box-shadow: none !important;
      border: none !important;
      padding: 0 !important;
      margin-top: 0 !important;
      margin-left: 0 !important;
      min-width: auto !important;
      opacity: 1 !important;
      visibility: visible !important;
      transform: none !important;
    }

    [data-label="Site Menu"] .dropdown-content .dropdown-item {
      padding: 0 !important;
      display: block;
    }

    [data-label="Site Menu"] .dropdown-content .dropdown-item a {
      font-size: 1.5rem;
      line-height: 2rem;
      padding: 0.5rem 0;
      color: rgba(255, 255, 255, 0.8);
      display: block;
    }

    [data-label="Site Menu"] .dropdown-content .dropdown-item a:hover {
      color: white;
      background-color: transparent;
    }
  }
</style>

<script>
  function initDropdowns() {
    // Organize sub-items into their parent dropdowns
    // Handle all navigation instances (desktop and mobile menu)
    const navContainers = document.querySelectorAll('[data-nav-items]');
    if (!navContainers || navContainers.length === 0) return;
    
    navContainers.forEach(function(navContainer) {
      const navItems = Array.from(navContainer.querySelectorAll('li'));
      const subItems = Array.from(navContainer.querySelectorAll('.dropdown-subitem'));
      
      // First, identify which nav items should have dropdowns (those with sub-items following them)
      const itemsWithDropdowns = new Set();
      
      subItems.forEach(function(subItem) {
        let prevSibling = subItem.previousElementSibling;
        // Look backwards to find the parent nav item
        while (prevSibling) {
          if (prevSibling.classList.contains('nav-item') && !prevSibling.classList.contains('dropdown-subitem')) {
            itemsWithDropdowns.add(prevSibling);
            break;
          }
          prevSibling = prevSibling.previousElementSibling;
        }
      });
      
      // Add dropdown functionality only to items that have sub-items
      itemsWithDropdowns.forEach(function(parentItem) {
        parentItem.classList.add('dropdown-parent');
        const link = parentItem.querySelector('a');
        if (link) {
          link.classList.add('dropbtn');
          // Add arrow SVG
          const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
          arrow.setAttribute('class', 'dropdown-arrow');
          arrow.setAttribute('width', '12');
          arrow.setAttribute('height', '12');
          arrow.setAttribute('viewBox', '0 0 12 12');
          arrow.setAttribute('fill', 'none');
          arrow.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
          const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
          path.setAttribute('d', 'M3 4.5L6 7.5L9 4.5');
          path.setAttribute('stroke', 'currentColor');
          path.setAttribute('stroke-width', '1.5');
          path.setAttribute('stroke-linecap', 'round');
          path.setAttribute('stroke-linejoin', 'round');
          arrow.appendChild(path);
          link.appendChild(arrow);
        }
      });
      
      // Process sub-items in reverse order to avoid index issues
      subItems.reverse().forEach(function(subItem) {
        if (!subItem.parentNode) return; // Already moved
        
        // Find the parent item that immediately precedes this sub-item
        let currentParent = null;
        let prevSibling = subItem.previousElementSibling;
        
        // Look backwards through siblings to find the most recent parent item
        while (prevSibling) {
          if (prevSibling.classList.contains('dropdown-parent')) {
            currentParent = prevSibling;
            break;
          }
          prevSibling = prevSibling.previousElementSibling;
        }
        
        if (currentParent) {
          const dropdown = currentParent.querySelector('.dropdown-content');
          if (dropdown) {
            // Clone and move the sub-item into the dropdown
            const clonedItem = subItem.cloneNode(true);
            clonedItem.classList.remove('dropdown-subitem');
            clonedItem.classList.add('dropdown-item');
            // Remove onclick from cloned link
            const clonedLink = clonedItem.querySelector('a');
            if (clonedLink) {
              clonedLink.removeAttribute('onclick');
              // Clean the label text (remove "- " prefix)
              const labelSpan = clonedLink.querySelector('span');
              if (labelSpan && labelSpan.textContent.trim().startsWith('-')) {
                labelSpan.textContent = labelSpan.textContent.trim().substring(1).trim();
              }
            }
            dropdown.appendChild(clonedItem);
            subItem.remove();
          }
        }
      });
    }); // End forEach navContainer
  }

  // Run immediately if DOM is ready, otherwise wait
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDropdowns);
  } else {
    initDropdowns();
  }

  // Re-initialize when mobile menu opens (in case it's rendered dynamically)
  document.addEventListener('DOMContentLoaded', function() {
    // Watch for mobile menu opening
    const mobileMenuTrigger = document.querySelector('[data-label="Site Menu"]');
    if (mobileMenuTrigger) {
      // Use MutationObserver to detect when mobile menu content is added
      const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
          if (mutation.addedNodes.length > 0) {
            // Check if navigation items were added
            const hasNavItems = Array.from(mutation.addedNodes).some(function(node) {
              return node.nodeType === 1 && (
                node.querySelector('[data-nav-items]') || 
                node.classList?.contains('dropdown-subitem') ||
                node.querySelector?.('.dropdown-subitem')
              );
            });
            if (hasNavItems) {
              setTimeout(initDropdowns, 50);
            }
          }
        });
      });
      
      observer.observe(mobileMenuTrigger, {
        childList: true,
        subtree: true
      });
    }
    
    // Also re-initialize on any click that might open mobile menu
    document.addEventListener('click', function(e) {
      if (e.target.closest('[data-label="Site Menu"]') || e.target.closest('.mobile-menu')) {
        setTimeout(initDropdowns, 100);
      }
    });
  });

  // Add mobile click support for dropdown arrows
  function setupMobileDropdowns() {
    const arrows = document.querySelectorAll('.dropdown-arrow');
    
    arrows.forEach(function(arrow) {
      // Skip if already has click handler
      if (arrow.dataset.clickSetup === 'true') return;
      arrow.dataset.clickSetup = 'true';
      
      const parent = arrow.closest('.dropdown-parent');
      if (!parent) return;
      
      const dropdown = parent.querySelector('.dropdown-content');
      if (!dropdown) return;
      
      // Make arrow clickable
      arrow.style.pointerEvents = 'auto';
      arrow.style.cursor = 'pointer';
      
      // Arrow click handler - toggle dropdown on mobile
      arrow.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        
        // Only work on mobile (max-width: 1023px)
        if (window.innerWidth > 1023) return;
        
        const isShowing = dropdown.classList.contains('show');
        
        // Close all other dropdowns first
        document.querySelectorAll('.dropdown-content.show').forEach(function(dd) {
          if (dd !== dropdown) {
            dd.classList.remove('show');
          }
        });
        document.querySelectorAll('.dropdown-arrow.show').forEach(function(arr) {
          if (arr !== arrow) {
            arr.classList.remove('show');
          }
        });
        
        // Toggle current dropdown
        if (isShowing) {
          dropdown.classList.remove('show');
          arrow.classList.remove('show');
        } else {
          dropdown.classList.add('show');
          arrow.classList.add('show');
        }
      });
    });
    
    // Close dropdowns when clicking outside (mobile only)
    if (!document.body.dataset.dropdownOutsideClick) {
      document.body.dataset.dropdownOutsideClick = 'true';
      document.addEventListener('click', function(event) {
        if (window.innerWidth > 1023) return;
        
        // If click is not on arrow or dropdown, close all
        if (!event.target.closest('.dropdown-arrow') && !event.target.closest('.dropdown-content')) {
          document.querySelectorAll('.dropdown-content.show').forEach(function(dd) {
            dd.classList.remove('show');
          });
          document.querySelectorAll('.dropdown-arrow.show').forEach(function(arr) {
            arr.classList.remove('show');
          });
        }
      });
    }
  }

  // Setup mobile dropdowns after initialization
  function initializeMobileDropdowns() {
    // Wait for dropdowns to be initialized by initDropdowns()
    setTimeout(setupMobileDropdowns, 150);
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeMobileDropdowns);
  } else {
    initializeMobileDropdowns();
  }
  
  // Re-run setup when mobile menu opens (dynamic content)
  document.addEventListener('click', function(e) {
    if (e.target.closest('[data-label="Site Menu"]') || e.target.closest('.mobile-menu')) {
      setTimeout(setupMobileDropdowns, 150);
    }
  });
</script>

  </div>

    <button class="absolute right-5 top-10 cursor-pointer w-8 focus:outline-none active:outline-none header-desktop-search" data-ghost-search title="Search" aria-label="Search">
      <i class="icon icon-search w-6 h-6">
  <svg xmlns="http://www.w3.org/2000/svg" class="fill-current w-6" viewBox="0 0 35 35">
  <path d="M34.464 31.487l-7.935-7.934a14.656 14.656 0 002.938-8.82C29.467 6.61 22.857 0 14.734 0 6.61 0 0 6.61 0 14.734c0 8.124 6.61 14.733 14.734 14.733a14.656 14.656 0 008.818-2.938l7.935 7.935a2.109 2.109 0 002.977-2.977zM4.21 14.734a10.524 10.524 0 1110.524 10.524A10.537 10.537 0 014.21 14.734z"></path>
</svg>
</i>    </button>
</header>

<header
    x-data="{ isSticky: false }"
    x-init="window.addEventListener('scroll', () => { isSticky = window.pageYOffset > 0 })"
    class="sticky top-0 left-0 z-503 lg:hidden transition-colors duration-200 border-b"
    :class="{
        'bg-black border-white/25': showMenu,
        'bg-body-bg border-black/25  ': !showMenu,
        'menu-is-sticky': isSticky
    }"
>
  <div class="relative flex flex-row items-center justify-center w-full px-6 py-4 h-[68px]">
    <div class="absolute h-full w-[68px] left-0 flex items-center cursor-pointer justify-center transition-colors duration-200"
         :class="{ 'border-r border-white/25': showMenu, 'border-r border-black/25 ': !showMenu }">
      <button
        @click="showMenu = !showMenu"
        class="w-6 h-6 cursor-pointer "
        :class="{ 'text-white': showMenu, 'text-black': !showMenu }"
        aria-label="Toggle Menu"
      >
        <svg x-show="!showMenu" class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path>
        </svg>
        <svg x-show="showMenu" x-cloak class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
          <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
        </svg>
      </button>
    </div>

    <a class="flex items-center fr-logo" href="https://review.firstround.com" aria-label="First Round">
      <div x-show="!showMenu">
        <svg width="38" height="76" viewBox="0 0 38 76" class="fill-current h-[48px]" fill="none" xmlns="http://www.w3.org/2000/svg">
<g clip-path="url(#clip0_635_302)">
<path d="M4.35052 0L0 19.8845H12.4484V76H37.4178V0H4.35052Z" fill="#000503"/>
<path d="M29.7616 61.1697V64.3195L25.3385 66.7489V67.6034H29.7616V70.3176H18.2012V66.1625C18.2012 64.8724 18.4692 63.8672 19.0221 63.1132C19.5918 62.3425 20.5133 61.8399 21.6693 61.8399C23.8641 61.8399 24.7186 63.2473 24.9866 64.1017L29.7616 61.1697ZM20.597 65.1572C20.3457 65.4923 20.2955 65.9279 20.2955 66.5479V67.6201H23.261V66.5479C23.261 65.9279 23.1939 65.4923 22.9426 65.1572C22.6913 64.8054 22.373 64.5709 21.7698 64.5709C21.1667 64.5709 20.8483 64.8054 20.597 65.1572Z" fill="white"/>
<path d="M26.2773 52.0219V57.9027C27.4669 57.7854 28.003 56.8806 28.003 55.7916C28.003 54.7026 27.5339 53.7811 27.2155 53.312L28.6061 52.2565C28.8072 52.5245 29.9465 53.6136 29.9465 56.177C29.9465 58.7404 28.3716 60.5833 25.6406 60.5833C22.9097 60.5833 21.3516 58.5728 21.3516 56.1267C21.3516 53.6806 22.9767 51.9884 25.3893 51.9884C25.7914 51.9884 26.2773 52.0219 26.2773 52.0219ZM24.7527 54.4848C23.9987 54.5015 23.094 55.0544 23.094 56.177C23.094 57.2995 23.9652 57.8524 24.7527 57.8859V54.4848Z" fill="white"/>
<path d="M21.502 41.9023L29.7618 45.3035V47.8166L21.502 51.2009V48.403L26.7293 46.5265L21.502 44.6668V41.9023Z" fill="white"/>
<path d="M20.5468 39.2217C20.5468 40.1264 19.9939 40.7464 19.1562 40.7464C18.3185 40.7464 17.7656 40.1264 17.7656 39.2217C17.7656 38.317 18.3353 37.6971 19.1562 37.6971C19.9772 37.6971 20.5468 38.317 20.5468 39.2217ZM29.7617 37.8814V40.5621H21.5018V37.8814H29.7617Z" fill="white"/>
<path d="M26.2773 27.4937V33.3744C27.4669 33.2572 28.003 32.3524 28.003 31.2634C28.003 30.1744 27.5339 29.2529 27.2155 28.7838L28.6061 27.7283C28.8072 27.9963 29.9465 29.0854 29.9465 31.6488C29.9465 34.2122 28.3716 36.0551 25.6406 36.0551C22.9097 36.0551 21.3516 34.0446 21.3516 31.5985C21.3516 29.1524 22.9767 27.4602 25.3893 27.4602C25.7914 27.4602 26.2773 27.4937 26.2773 27.4937ZM24.7527 29.9566C23.9987 29.9733 23.094 30.5262 23.094 31.6488C23.094 32.7713 23.9652 33.3242 24.7527 33.3577V29.9566Z" fill="white"/>
<path d="M21.502 13.0181L29.7618 15.7323V18.2621L24.702 19.8705L29.7618 21.4957V24.0256L21.502 26.7398V24.0256L26.7795 22.568L21.502 20.9763V18.7815L26.7795 17.1899L21.502 15.7323V13.0181Z" fill="white"/>
</g>
<defs>
<clipPath id="clip0_635_302">
<rect width="37.4178" height="76" fill="white"/>
</clipPath>
</defs>
</svg>
      </div>
      <div x-show="showMenu" x-cloak>

      </div>
    </a>


    <button class="border-l border-black-100/25 focus:outline-none active:outline-none absolute h-full w-[68px] right-0 flex items-center cursor-pointer justify-center transition-colors duration-200 " data-ghost-search title="Search" aria-label="Search">
      <i class="icon icon-search w-5 h-5">
  <svg xmlns="http://www.w3.org/2000/svg" class="fill-current w-6" viewBox="0 0 35 35">
  <path d="M34.464 31.487l-7.935-7.934a14.656 14.656 0 002.938-8.82C29.467 6.61 22.857 0 14.734 0 6.61 0 0 6.61 0 14.734c0 8.124 6.61 14.733 14.734 14.733a14.656 14.656 0 008.818-2.938l7.935 7.935a2.109 2.109 0 002.977-2.977zM4.21 14.734a10.524 10.524 0 1110.524 10.524A10.537 10.537 0 014.21 14.734z"></path>
</svg>
</i>    </button>
  </div>
</header>

<div
    x-show="showMenu"
    x-transition:enter="transition ease-out duration-200"
    x-transition:enter-start="opacity-0 -translate-x-full"
    x-transition:enter-end="opacity-100 translate-x-0"
    x-transition:leave="transition ease-in duration-200"
    x-transition:leave-start="opacity-100 translate-x-0"
    x-transition:leave-end="opacity-0 -translate-x-full"
    x-cloak
    class="fixed inset-0 z-[502] lg:hidden"
>
    <div
        x-show="showMenu"
        x-transition:enter="transition-opacity ease-out duration-200"
        x-transition:enter-start="opacity-0"
        x-transition:enter-end="opacity-100"
        x-transition:leave="transition-opacity ease-in duration-200"
        x-transition:leave-start="opacity-100"
        x-transition:leave-end="opacity-0"
        @click="showMenu = false"
        class="fixed inset-0 bg-black bg-opacity-50"
        x-cloak
    ></div>

    <nav
        class="fixed top-0 bottom-0 left-0 w-full bg-black overflow-y-auto px-8 pt-[80px] transform"
        @click.away="showMenu = false"
    >
        <div class="flex justify-end">
          <svg width="89" height="167" class="fill-current h-[167px] text-white" viewBox="0 0 89 167" fill="none" xmlns="http://www.w3.org/2000/svg">
<path fill-rule="evenodd" clip-rule="evenodd" d="M13.9543 0L4.75977 43.6882H31.0559V167H83.8074V0H13.9543ZM40.5425 155.352H55.5266V151.757H49.4981V147.061L46.698 146.41V151.757H43.3438V145.728L40.5425 145.077V155.352ZM48.0216 130.361L44.9627 129.691V129.693C44.9627 129.693 44.6332 130.281 44.6332 131.122C44.6332 132.305 45.1756 133.221 46.3474 133.846V133.891L44.8274 133.868V137.44H55.5348V133.868H50.7793C49.5852 133.868 48.8899 133.578 48.5428 133.309C48.1957 133.041 47.7604 132.461 47.7604 131.657C47.7604 131.031 47.9122 130.562 48.0216 130.361ZM54.8346 121.404C54.2052 120.6 53.337 120.108 52.2287 120.108C51.1204 120.108 50.4051 120.644 50.0568 121.002C49.7086 121.382 49.4274 121.94 49.2533 122.431C49.0792 122.9 48.9697 123.235 48.7533 123.994C48.5368 124.754 48.2756 125.602 47.712 125.602C47.1485 125.602 46.9297 125.132 46.9297 124.284C46.9297 123.391 47.6262 121.899 48.1838 121.224L45.6967 120.67C45.2096 121.376 44.6061 122.609 44.6061 124.575C44.6061 126.875 45.8873 128.75 47.9732 128.75C49.8851 128.75 50.558 127.388 50.7745 126.875C50.991 126.361 51.1663 125.737 51.2745 125.312C51.2921 125.258 51.3136 125.182 51.3394 125.091C51.5049 124.507 51.8459 123.303 52.4475 123.303C53.1428 123.303 53.3593 123.929 53.3593 124.821C53.3593 125.714 52.7946 127.189 52.0557 128.081L54.1193 129.422C55.2711 128.171 55.77 126.43 55.77 124.777C55.77 123.303 55.4441 122.208 54.8358 121.405L54.8346 121.404ZM47.4666 113.908V111.75H47.4677L44.8394 111.177V113.908H41.9287V117.503H44.8371V119.066H47.4654V117.503H51.6138C53.2421 117.503 54.1327 117.279 54.7621 116.721C55.3939 116.163 55.7622 115.336 55.7622 114.042C55.7622 112.864 55.5283 111.944 55.4375 111.586C55.4285 111.551 55.4209 111.521 55.4151 111.497L52.6997 111.676C52.6997 111.676 52.895 112.189 52.895 112.748C52.895 113.217 52.8091 113.44 52.635 113.596C52.4397 113.775 52.222 113.908 51.3102 113.908H47.4666ZM63.264 144.054C66.1074 144.054 67.216 145.928 67.5636 147.067L73.7538 143.16V147.358L68.0194 150.596V151.735H73.7538V155.352H58.7674V149.816C58.7674 148.097 59.1156 146.756 59.8321 145.751C60.5709 144.724 61.7663 144.054 63.264 144.054ZM64.9158 148.474C65.2417 148.921 65.3288 149.501 65.3288 150.327V151.756H61.4851V150.327C61.4851 149.501 61.5498 148.921 61.8769 148.474C62.2028 148.006 62.6369 147.693 63.3969 147.693C64.157 147.693 64.5899 148.006 64.917 148.474H64.9158ZM68.4247 142.876C64.9728 142.876 62.8445 140.242 62.8445 136.758C62.8445 133.274 64.9705 130.661 68.4247 130.661C71.8789 130.661 73.9849 133.274 73.9849 136.758C73.9849 140.242 71.8766 142.876 68.4247 142.876ZM68.36 139.258C69.9459 139.258 71.1401 138.344 71.1401 136.758C71.1401 135.195 69.9447 134.257 68.36 134.257C66.7752 134.257 65.5799 135.173 65.5799 136.758C65.5799 138.342 66.7741 139.258 68.36 139.258ZM73.7422 118.269V121.708L72.5257 121.73C73.1775 122.199 73.9799 123.361 73.9799 124.945C73.9799 126.53 73.4375 127.535 72.721 128.094C71.9821 128.63 71.048 128.898 69.5068 128.898H63.0348V125.326H69.0291C70.2656 125.326 71.1139 124.968 71.1139 123.696C71.1139 122.423 70.1362 121.841 68.7244 121.841H63.0348V118.269H73.7422ZM73.7484 108.584V105.012H73.7472H67.4282C64.8434 105.012 62.8445 105.837 62.8445 109.03C62.8445 110.616 63.6281 111.621 64.5822 112.224V112.268L63.041 112.245V115.818H73.7484V112.245H68.4494C66.907 112.245 65.6905 111.754 65.6905 110.281C65.6905 108.808 66.7541 108.584 68.1659 108.584H73.7484ZM58.7674 94.5529H64.3487C63.8264 94.911 62.8498 96.0263 62.8498 98.0368C62.8498 101.23 65.2605 103.24 68.41 103.24C71.5596 103.24 73.9902 101.23 73.9902 98.0368C73.9902 96.5634 73.4479 95.2461 72.3619 94.4199L72.3408 94.3533L73.7526 94.3763V90.9832L58.7674 90.9832V94.5529ZM71.1466 97.0098C71.1466 98.5497 70.0172 99.5997 68.3665 99.5997C66.7159 99.5997 65.5864 98.5497 65.5864 97.0098C65.5864 95.4699 66.7159 94.4416 68.3665 94.4416C70.0172 94.4416 71.1466 95.4687 71.1466 97.0098ZM47.61 144.555L44.8099 143.932V139.475H55.5162V143.047H47.61V144.555Z" fill="currentColor"/>
</svg>
        </div>
        <div data-label="Site Menu" class="">
  <div class="flex flex-col gap-8 py-8 ">
    <ul class="flex flex-col divide-y divide-gray-100/20 max-w-[266px] w-full" data-nav-items>
      <li class="nav-articles py-6 px-0 text-left nav-item"
          data-label="Articles" data-length="8" data-slug="articles">
        <a class="font-skandia-bold text-3xl leading-[44.8px] tracking-[-0.03em] cursor-pointer text-white" href="https://review.firstround.com/articles/">
          <span>Articles</span>
        </a>
        <div id="tag-content-articles" class="dropdown-content"></div>
      </li>
      <li class="nav-curated-reads py-6 px-0 text-left nav-item"
          data-label="Curated Reads" data-length="13" data-slug="curated-reads">
        <a class="font-skandia-bold text-3xl leading-[44.8px] tracking-[-0.03em] cursor-pointer text-white" href="https://review.firstround.com/collections/">
          <span>Curated Reads</span>
        </a>
        <div id="tag-content-curated-reads" class="dropdown-content"></div>
      </li>
      <li class="nav-paths-to-pmf py-6 px-0 text-left nav-item"
          data-label="Paths to PMF" data-length="12" data-slug="paths-to-pmf">
        <a class="font-skandia-bold text-3xl leading-[44.8px] tracking-[-0.03em] cursor-pointer text-white" href="https://review.firstround.com/series/product-market-fit/">
          <span>Paths to PMF</span>
        </a>
        <div id="tag-content-paths-to-pmf" class="dropdown-content"></div>
      </li>
      <li class="nav-newsletter py-6 px-0 text-left nav-item"
          data-label="Newsletter" data-length="10" data-slug="newsletter">
        <a class="font-skandia-bold text-3xl leading-[44.8px] tracking-[-0.03em] cursor-pointer text-white" href="https://review.firstround.com/newsletter/">
          <span>Newsletter</span>
        </a>
        <div id="tag-content-newsletter" class="dropdown-content"></div>
      </li>
      <li class="nav-podcast py-6 px-0 text-left nav-item"
          data-label="Podcast" data-length="7" data-slug="podcast">
        <a class="font-skandia-bold text-3xl leading-[44.8px] tracking-[-0.03em] cursor-pointer text-white" href="https://review.firstround.com/podcast/">
          <span>Podcast</span>
        </a>
        <div id="tag-content-podcast" class="dropdown-content"></div>
      </li>
      <li class="nav-executive-function py-6 px-0 text-left is-subitem dropdown-subitem" 
          data-label="-Executive Function" data-length="19">
        <a class="font-skandia-bold text-3xl leading-[44.8px] tracking-[-0.03em] cursor-pointer text-white" href="https://review.firstround.com/executive-function/">
          <span>-Executive Function</span>
        </a>
      </li>
      <li class="nav-in-depth py-6 px-0 text-left is-subitem dropdown-subitem" 
          data-label="-In Depth" data-length="9">
        <a class="font-skandia-bold text-3xl leading-[44.8px] tracking-[-0.03em] cursor-pointer text-white" href="https://review.firstround.com/podcast/">
          <span>-In Depth</span>
        </a>
      </li>

     <li class="nav- nav-current  px-4 flex items-center max-lg:py-6 max-lg:px-0 max-lg:text-left fr-signin"
      data-label="" data-length="">
      <a href="/signin/" class="text-sm lg:px-2 leading-none h-[30px] flex items-center justify-center max-lg:font-skandia-bold max-lg:text-3xl max-lg:leading-[44.8px] max-lg:tracking-[-0.03em] max-lg:cursor-pointer max-lg:text-white" data-portal="signin"><span>Sign In</span></a>
    </li>
</ul>

<style>
  /* Dropdown parent - maintain existing layout */
  .dropdown-parent {
    position: relative;
  }

  /* Dropdown arrow indicator */
  .dropdown-parent .dropbtn {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
  }

  .dropdown-parent .dropdown-arrow {
    opacity: 0.6;
    transition: transform 0.3s ease, opacity 0.3s ease;
    margin-left: 0.25rem;
    flex-shrink: 0;
  }

  /* Dropdown content - positioned absolutely, doesn't affect layout */
  .dropdown-content {
    display: none;
    position: absolute;
    top: 100%;
    left: 0;
    background-color: var(--color-body-bg, #fff);
    min-width: 200px;
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    z-index: 1000;
    padding: 0.5rem 6px;
    margin-top: 0.5rem;
    border-radius: 0.25rem;
    border: 1px solid rgba(0, 0, 0, 0.05);
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.3s ease, visibility 0.3s ease, transform 0.3s ease;
    transform: translateY(-10px);
  }

  /* Desktop: Show dropdown on hover */
  @media (min-width: 1024px) {
    .dropdown-parent:hover .dropdown-arrow {
      opacity: 1;
      transform: rotate(180deg);
    }

    .dropdown-parent:hover .dropdown-content {
      display: block;
      opacity: 1;
      visibility: visible;
      transform: translateY(0);
      top: 48px;
    }
  }

  .dropdown-content.show {
    display: block;
    opacity: 1;
    visibility: visible;
    transform: translateY(0);
  }

  /* Dropdown items - match theme styling */
  .dropdown-content .dropdown-item {
    display: block;
    transition: opacity 0.3s ease;
    padding: 0 6px;
  }

  .dropdown-content .dropdown-item a {
    display: block;
    padding: 0.625rem 0.625rem;
    text-decoration: none;
    color: var(--color-typography, #000);
    font-family: var(--font-body);
    font-size: 0.875rem;
    transition: all 0.3s ease;
  }

  .dropdown-content .dropdown-item a:hover {
    background-color: rgba(0, 0, 0, 0.03);
    font-weight: 500;
  }

  /* Mobile: dropdown items should look like regular menu items */
  @media (max-width: 1023px) {
    .dropdown-content .dropdown-item {
      padding: 0 !important;
      background-color: transparent !important;
    }

    .dropdown-content .dropdown-item a {
      background-color: transparent !important;
      padding: 0.5rem 0 !important;
    }

    .dropdown-content .dropdown-item a:hover {
      background-color: transparent !important;
    }
  }

  /* Mobile: Disable hover completely, only allow click on arrow */
  @media (max-width: 1023px) {
    /* Disable ALL hover effects on mobile - prevent any hover state */
    .dropdown-parent:hover .dropdown-content.show {
      display: block !important;
      opacity: 1 !important;
      visibility: visible !important;
      transform: none !important;
      transition: none !important;
      pointer-events: auto !important;
    }

    .dropdown-parent:hover .dropdown-arrow {
      transform: rotate(-90deg) !important;
      opacity: 0.8 !important;
      transition: none !important;
    }
  }

  /* Touch devices: Completely disable hover (more specific than max-width) */
  @media (hover: none) and (pointer: coarse) {
    .dropdown-parent:hover .dropdown-content {
      display: none !important;
      opacity: 0 !important;
      visibility: hidden !important;
      transform: none !important;
      transition: none !important;
      pointer-events: none !important;
    }

    .dropdown-parent:hover .dropdown-arrow {
      transform: rotate(-90deg) !important;
      opacity: 0.8 !important;
      transition: none !important;
    }
  }

  @media (max-width: 1023px) {
    .dropdown-content {
      position: static !important;
      background-color: transparent !important;
      box-shadow: none !important;
      border: none !important;
      margin-top: 0 !important;
      padding-left: 1rem;
      padding-right: 0 !important;
      transform: none !important;
      transition: none !important;
      display: none !important;
      opacity: 1 !important;
      visibility: hidden !important;
      min-width: auto !important;
      pointer-events: none !important;
    }

    /* Only show when .show class is added via JS click */
    .dropdown-content.show {
      display: block !important;
      opacity: 1 !important;
      visibility: visible !important;
      pointer-events: auto !important;
    }

    .dropdown-parent .dropdown-arrow {
      transform: rotate(-90deg);
      cursor: pointer;
      pointer-events: auto !important;
      opacity: 0.8;
      min-width: 32px;
      min-height: 32px;
      padding: 8px;
      margin-left: 0.5rem;
      display: inline-flex !important;
      align-items: center;
      justify-content: center;
      -webkit-tap-highlight-color: transparent;
      touch-action: manipulation;
    }

    .dropdown-parent .dropdown-arrow:active {
      opacity: 1;
    }

    .dropdown-parent .dropdown-arrow.show {
      transform: rotate(90deg);
      opacity: 1;
    }

    /* Mobile menu specific styling */
    [data-label="Site Menu"] .dropdown-parent .dropbtn {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
    }

    [data-label="Site Menu"] .dropdown-content {
      position: static !important;
      background-color: transparent !important;
      box-shadow: none !important;
      border: none !important;
      padding: 0 !important;
      margin-top: 0 !important;
      margin-left: 0 !important;
      min-width: auto !important;
      opacity: 1 !important;
      visibility: visible !important;
      transform: none !important;
    }

    [data-label="Site Menu"] .dropdown-content .dropdown-item {
      padding: 0 !important;
      display: block;
    }

    [data-label="Site Menu"] .dropdown-content .dropdown-item a {
      font-size: 1.5rem;
      line-height: 2rem;
      padding: 0.5rem 0;
      color: rgba(255, 255, 255, 0.8);
      display: block;
    }

    [data-label="Site Menu"] .dropdown-content .dropdown-item a:hover {
      color: white;
      background-color: transparent;
    }
  }
</style>

<script>
  function initDropdowns() {
    // Organize sub-items into their parent dropdowns
    // Handle all navigation instances (desktop and mobile menu)
    const navContainers = document.querySelectorAll('[data-nav-items]');
    if (!navContainers || navContainers.length === 0) return;
    
    navContainers.forEach(function(navContainer) {
      const navItems = Array.from(navContainer.querySelectorAll('li'));
      const subItems = Array.from(navContainer.querySelectorAll('.dropdown-subitem'));
      
      // First, identify which nav items should have dropdowns (those with sub-items following them)
      const itemsWithDropdowns = new Set();
      
      subItems.forEach(function(subItem) {
        let prevSibling = subItem.previousElementSibling;
        // Look backwards to find the parent nav item
        while (prevSibling) {
          if (prevSibling.classList.contains('nav-item') && !prevSibling.classList.contains('dropdown-subitem')) {
            itemsWithDropdowns.add(prevSibling);
            break;
          }
          prevSibling = prevSibling.previousElementSibling;
        }
      });
      
      // Add dropdown functionality only to items that have sub-items
      itemsWithDropdowns.forEach(function(parentItem) {
        parentItem.classList.add('dropdown-parent');
        const link = parentItem.querySelector('a');
        if (link) {
          link.classList.add('dropbtn');
          // Add arrow SVG
          const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
          arrow.setAttribute('class', 'dropdown-arrow');
          arrow.setAttribute('width', '12');
          arrow.setAttribute('height', '12');
          arrow.setAttribute('viewBox', '0 0 12 12');
          arrow.setAttribute('fill', 'none');
          arrow.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
          const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
          path.setAttribute('d', 'M3 4.5L6 7.5L9 4.5');
          path.setAttribute('stroke', 'currentColor');
          path.setAttribute('stroke-width', '1.5');
          path.setAttribute('stroke-linecap', 'round');
          path.setAttribute('stroke-linejoin', 'round');
          arrow.appendChild(path);
          link.appendChild(arrow);
        }
      });
      
      // Process sub-items in reverse order to avoid index issues
      subItems.reverse().forEach(function(subItem) {
        if (!subItem.parentNode) return; // Already moved
        
        // Find the parent item that immediately precedes this sub-item
        let currentParent = null;
        let prevSibling = subItem.previousElementSibling;
        
        // Look backwards through siblings to find the most recent parent item
        while (prevSibling) {
          if (prevSibling.classList.contains('dropdown-parent')) {
            currentParent = prevSibling;
            break;
          }
          prevSibling = prevSibling.previousElementSibling;
        }
        
        if (currentParent) {
          const dropdown = currentParent.querySelector('.dropdown-content');
          if (dropdown) {
            // Clone and move the sub-item into the dropdown
            const clonedItem = subItem.cloneNode(true);
            clonedItem.classList.remove('dropdown-subitem');
            clonedItem.classList.add('dropdown-item');
            // Remove onclick from cloned link
            const clonedLink = clonedItem.querySelector('a');
            if (clonedLink) {
              clonedLink.removeAttribute('onclick');
              // Clean the label text (remove "- " prefix)
              const labelSpan = clonedLink.querySelector('span');
              if (labelSpan && labelSpan.textContent.trim().startsWith('-')) {
                labelSpan.textContent = labelSpan.textContent.trim().substring(1).trim();
              }
            }
            dropdown.appendChild(clonedItem);
            subItem.remove();
          }
        }
      });
    }); // End forEach navContainer
  }

  // Run immediately if DOM is ready, otherwise wait
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDropdowns);
  } else {
    initDropdowns();
  }

  // Re-initialize when mobile menu opens (in case it's rendered dynamically)
  document.addEventListener('DOMContentLoaded', function() {
    // Watch for mobile menu opening
    const mobileMenuTrigger = document.querySelector('[data-label="Site Menu"]');
    if (mobileMenuTrigger) {
      // Use MutationObserver to detect when mobile menu content is added
      const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
          if (mutation.addedNodes.length > 0) {
            // Check if navigation items were added
            const hasNavItems = Array.from(mutation.addedNodes).some(function(node) {
              return node.nodeType === 1 && (
                node.querySelector('[data-nav-items]') || 
                node.classList?.contains('dropdown-subitem') ||
                node.querySelector?.('.dropdown-subitem')
              );
            });
            if (hasNavItems) {
              setTimeout(initDropdowns, 50);
            }
          }
        });
      });
      
      observer.observe(mobileMenuTrigger, {
        childList: true,
        subtree: true
      });
    }
    
    // Also re-initialize on any click that might open mobile menu
    document.addEventListener('click', function(e) {
      if (e.target.closest('[data-label="Site Menu"]') || e.target.closest('.mobile-menu')) {
        setTimeout(initDropdowns, 100);
      }
    });
  });

  // Add mobile click support for dropdown arrows
  function setupMobileDropdowns() {
    const arrows = document.querySelectorAll('.dropdown-arrow');
    
    arrows.forEach(function(arrow) {
      // Skip if already has click handler
      if (arrow.dataset.clickSetup === 'true') return;
      arrow.dataset.clickSetup = 'true';
      
      const parent = arrow.closest('.dropdown-parent');
      if (!parent) return;
      
      const dropdown = parent.querySelector('.dropdown-content');
      if (!dropdown) return;
      
      // Make arrow clickable
      arrow.style.pointerEvents = 'auto';
      arrow.style.cursor = 'pointer';
      
      // Arrow click handler - toggle dropdown on mobile
      arrow.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        
        // Only work on mobile (max-width: 1023px)
        if (window.innerWidth > 1023) return;
        
        const isShowing = dropdown.classList.contains('show');
        
        // Close all other dropdowns first
        document.querySelectorAll('.dropdown-content.show').forEach(function(dd) {
          if (dd !== dropdown) {
            dd.classList.remove('show');
          }
        });
        document.querySelectorAll('.dropdown-arrow.show').forEach(function(arr) {
          if (arr !== arrow) {
            arr.classList.remove('show');
          }
        });
        
        // Toggle current dropdown
        if (isShowing) {
          dropdown.classList.remove('show');
          arrow.classList.remove('show');
        } else {
          dropdown.classList.add('show');
          arrow.classList.add('show');
        }
      });
    });
    
    // Close dropdowns when clicking outside (mobile only)
    if (!document.body.dataset.dropdownOutsideClick) {
      document.body.dataset.dropdownOutsideClick = 'true';
      document.addEventListener('click', function(event) {
        if (window.innerWidth > 1023) return;
        
        // If click is not on arrow or dropdown, close all
        if (!event.target.closest('.dropdown-arrow') && !event.target.closest('.dropdown-content')) {
          document.querySelectorAll('.dropdown-content.show').forEach(function(dd) {
            dd.classList.remove('show');
          });
          document.querySelectorAll('.dropdown-arrow.show').forEach(function(arr) {
            arr.classList.remove('show');
          });
        }
      });
    }
  }

  // Setup mobile dropdowns after initialization
  function initializeMobileDropdowns() {
    // Wait for dropdowns to be initialized by initDropdowns()
    setTimeout(setupMobileDropdowns, 150);
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeMobileDropdowns);
  } else {
    initializeMobileDropdowns();
  }
  
  // Re-run setup when mobile menu opens (dynamic content)
  document.addEventListener('click', function(e) {
    if (e.target.closest('[data-label="Site Menu"]') || e.target.closest('.mobile-menu')) {
      setTimeout(setupMobileDropdowns, 150);
    }
  });
</script>
  </div>
</div>
    </nav>
</div>


    <main class="main">
      
  <div data-label="Article Cover" class="relative w-full max-w-7xl m-auto mb-5 max-lg:gap-0 lg:mb-10  flex flex-col items-center lg:gap-8 max-w-7xl w-full mx-auto mt-5 lg:mt-8 mb-5 lg:mb-12 px-5 lg:px-0 md:gap-12 lg:gap-16 lg:flex-row lg:flex-row-reverse">

    <div class="flex justify-center transform-none mb-5 lg:mb-10 flex-1 p-0!">
        <div data-label="Article Info" class="max-w-xl sm:max-w-4xl  2xl:max-w-6xl">
            <div class="max-w-[843px] w-full">
                  <a href="/articles/pr-and-marketing/" data-label="PrimaryCategory" class="firstround-tag">PR &amp; Marketing</a>
                <h1 data-label="Article Title" class="text-[40px] lg:text-5xl tracking-tight !font-skandia-medium mt-5 lg:mt-8 leading-[1.15] mb-3">Why Most Startups Don&#x27;t &#x27;Get&#x27; Press</h1>
                <p data-label="Description" class="text-2xl lg:text-3xl font-leif tracking-tight">PR guru Brooke Hammerling gives step-by-step advice for startups looking to get good press coverage without an agency.</p>

            </div>
        </div>
    </div>

          <figure class="aspect-auto max-w-full min-lg:max-w-[1200px] max-lg:order-3 m-auto flex-1 lg:aspect-16/9!">
            <picture  >
  <source 
    srcset="/content/images/size/w320/format/webp/2056/firstround/lwcokxqmrrcvhenijwy0_54832618_thumbnail.jpg 320w, /content/images/size/w640/format/webp/2056/firstround/lwcokxqmrrcvhenijwy0_54832618_thumbnail.jpg 600w, /content/images/size/w960/format/webp/2056/firstround/lwcokxqmrrcvhenijwy0_54832618_thumbnail.jpg 960w, /content/images/size/w1280/format/webp/2056/firstround/lwcokxqmrrcvhenijwy0_54832618_thumbnail.jpg 1200w, /content/images/size/w2000/format/webp/2056/firstround/lwcokxqmrrcvhenijwy0_54832618_thumbnail.jpg 2000w"
    sizes="100vw"
    type="image/webp"
  >
  <img class="w-full object-cover object-center lg:mt-0"
    
    loading="eager"
    srcset="/content/images/size/w320/format/webp/2056/firstround/lwcokxqmrrcvhenijwy0_54832618_thumbnail.jpg 320w, /content/images/size/w640/format/webp/2056/firstround/lwcokxqmrrcvhenijwy0_54832618_thumbnail.jpg 600w, /content/images/size/w960/format/webp/2056/firstround/lwcokxqmrrcvhenijwy0_54832618_thumbnail.jpg 960w, /content/images/size/w1280/format/webp/2056/firstround/lwcokxqmrrcvhenijwy0_54832618_thumbnail.jpg 1200w, /content/images/size/w2000/format/webp/2056/firstround/lwcokxqmrrcvhenijwy0_54832618_thumbnail.jpg 2000w"
    sizes="100vw"
    src="/content/images/size/w30/2056/firstround/lwcokxqmrrcvhenijwy0_54832618_thumbnail.jpg"
    alt="Why Most Startups Don&#x27;t &#x27;Get&#x27; Press"
  />
</picture>          </figure>


    <div class="mt-0 block self-start lg:hidden "><div class="mb-6 lg:mb-10 "
    x-data="authorComponent($el)"
    data-authors='[{"id": 0, "name": "First Round Staff", "url": "/404/", "profile_image": ""}]'
>
    <template x-if="secondaryAuthors.length > 0">
        <div class="post-author__meta flex gap-2 items-center mt-4">
            <div class="author-images flex">
                <!-- Display the first two authors' images -->
                <template x-for="(author, index) in secondaryAuthors.slice(0, 2)" :key="author.id">
                    <span>
                        <img :src="author.profile_image" :alt="author.name" class="h-[60px] w-[60px] rounded-full border-2 border-white object-cover">
                    </span>
                </template>

                <!-- If there are more than 2 authors, display +X circle -->
                <template x-if="secondaryAuthors.length > 2">
                    <div class="h-[60px] w-[60px] flex items-center justify-center rounded-full border-2 border-white bg-[#e9e9e9] text-gray-400 text-[28px] -ml-4">
                        +
                    </div>
                </template>
            </div>

            <div class="author-names flex flex-col">
                <span class="text-[11px] uppercase mb-[3px]">Featuring</span>
                <span class="text-base">
                    <span
                        x-text="secondaryAuthors.length === 1
                            ? secondaryAuthors[0].name
                            : secondaryAuthors.length === 2
                                ? `${secondaryAuthors[0].name} and ${secondaryAuthors[1].name}`
                                : secondaryAuthors.length === 3
                                    ? `${secondaryAuthors[0].name} and ${secondaryAuthors.length - 1} more`
                                    : `${secondaryAuthors[0].name} and more`">
                    </span>
                </span>
            </div>
        </div>
    </template>
</div>

<script>
document.addEventListener('alpine:init', () => {
    Alpine.data('authorComponent', (el) => ({
        authors: JSON.parse(el.dataset.authors),  // Get authors data from Handlebars
        get secondaryAuthors() {
            return this.authors.slice(1); // Ignore the primary author
        }
    }));
});
</script>
</div>


</div>

<div class="max-w-content m-auto w-full max-lg:px-5">

  <div class="border-b border-gray-100 flex items-center justify-between pb-5 lg:pb-6 mb-10 lg:mb-15 w-full max-w-[665px] m-auto">
  
    <div class="mb-6 lg:mb-10 hidden lg:block mb-0 lg:!mb-0"
    x-data="authorComponent($el)"
    data-authors='[{"id": 0, "name": "First Round Staff", "url": "/404/", "profile_image": ""}]'
>
    <template x-if="secondaryAuthors.length > 0">
        <div class="post-author__meta flex gap-2 items-center mt-4">
            <div class="author-images flex">
                <!-- Display the first two authors' images -->
                <template x-for="(author, index) in secondaryAuthors.slice(0, 2)" :key="author.id">
                    <span>
                        <img :src="author.profile_image" :alt="author.name" class="h-[60px] w-[60px] rounded-full border-2 border-white object-cover">
                    </span>
                </template>

                <!-- If there are more than 2 authors, display +X circle -->
                <template x-if="secondaryAuthors.length > 2">
                    <div class="h-[60px] w-[60px] flex items-center justify-center rounded-full border-2 border-white bg-[#e9e9e9] text-gray-400 text-[28px] -ml-4">
                        +
                    </div>
                </template>
            </div>

            <div class="author-names flex flex-col">
                <span class="text-[11px] uppercase mb-[3px]">Featuring</span>
                <span class="text-base">
                    <span
                        x-text="secondaryAuthors.length === 1
                            ? secondaryAuthors[0].name
                            : secondaryAuthors.length === 2
                                ? `${secondaryAuthors[0].name} and ${secondaryAuthors[1].name}`
                                : secondaryAuthors.length === 3
                                    ? `${secondaryAuthors[0].name} and ${secondaryAuthors.length - 1} more`
                                    : `${secondaryAuthors[0].name} and more`">
                    </span>
                </span>
            </div>
        </div>
    </template>
</div>

<script>
document.addEventListener('alpine:init', () => {
    Alpine.data('authorComponent', (el) => ({
        authors: JSON.parse(el.dataset.authors),  // Get authors data from Handlebars
        get secondaryAuthors() {
            return this.authors.slice(1); // Ignore the primary author
        }
    }));
});
</script>
  
    <div class="flex flex-row justify-center md:flex xl:col-span-8" data-share>
    <div data-label="Share Article" class="flex flex-row space-y-0 sm:space-y-12 items-center">
      <div class="flex flex-row items-center space-x-2 sm:space-x-2">
        <div data-label="Share Options" class="flex flex-row space-x-2 sm:space-x-2">
          <a class="flex flex-row justify-center items-center p-3 ibtn transition duration-200 rounded-full outline-hidden focus:outline-hidden hover:outline-hidden active:outline-hidden border border-black/25 h-[40px] w-[40px] text-black"  target="_blank"
            href="https://twitter.com/share?text=Why%20Most%20Startups%20Don't%20'Get'%20Press&amp;url=https://review.firstround.com/why-most-startups-dont-get-press/"
            title="Share on Twitter" aria-label="Tweet">
          <i class="icon icon-twitter-x stroke-1.5">
  <svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-brand-x hover:contrast-90 hover:brightness-110" width="16" height="16" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
  <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
  <path d="M4 4l11.733 16h4.267l-11.733 -16z"></path>
  <path d="M4 20l6.768 -6.768m2.46 -2.46l6.772 -6.772"></path>
</svg>
</i>          </a>
          <a class="flex flex-row justify-center items-center p-3 ibtn rounded-full transition duration-200 outline-hidden focus:outline-hidden hover:outline-hidden active:outline-hidden border border-black/25 h-[40px] w-[40px] text-black"target="_blank"
            href="https://www.linkedin.com/shareArticle?mini=true&url=https://review.firstround.com/why-most-startups-dont-get-press/&title=Why%20Most%20Startups%20Don't%20'Get'%20Press&summary=Why%20Most%20Startups%20Don't%20'Get'%20Press"
            title="Share on Linkedin" aria-label="Share">
            <i class="icon icon-linkedin stroke-1.5">
  <svg width="14" height="13" viewBox="0 0 14 13" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M9.59995 4.28955C10.6343 4.28955 11.6263 4.6888 12.3577 5.39946C13.0891 6.11013 13.5 7.07399 13.5 8.07902V12.5001H10.9V8.07902C10.9 7.74401 10.763 7.42273 10.5192 7.18584C10.2754 6.94895 9.94473 6.81587 9.59995 6.81587C9.25517 6.81587 8.92451 6.94895 8.68071 7.18584C8.43692 7.42273 8.29995 7.74401 8.29995 8.07902V12.5001H5.69995V8.07902C5.69995 7.07399 6.11084 6.11013 6.84223 5.39946C7.57363 4.6888 8.56561 4.28955 9.59995 4.28955Z" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M3.1 4.9209H0.5V12.4998H3.1V4.9209Z" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M1.8 3.02632C2.51797 3.02632 3.1 2.46078 3.1 1.76316C3.1 1.06554 2.51797 0.5 1.8 0.5C1.08203 0.5 0.5 1.06554 0.5 1.76316C0.5 2.46078 1.08203 3.02632 1.8 3.02632Z" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/>
</svg>

</i>          </a>
        </div>
      </div>
    </div>
  </div>
</div>
</div>

  <div data-label="Article Content" class="lg:grid lg:grid-cols-12 mt-0 lg:mt-[2rem]" x-data="{showToc : true}">
    <div data-label="Article Sidebar"
    x-data="{
      showToc: false,
      isMobile: window.innerWidth < 1024
    }"
    x-effect="showToc ? document.body.classList.add('toc-open') : document.body.classList.remove('toc-open')"
    x-init="
      init();
      window.addEventListener('resize', () => {
        isMobile = window.innerWidth < 1024;
      });
    "
    class="col-span-0 lg:h-screen-minus-header max-lg:pl-5 lg:sticky top-[8rem] lg:left-[7rem] lg:top-[7.5rem] lg:h-screen-minus-header sm:col-span-3 lg:col-span-2 lg:h-[70vh] max-lg:mb-4 z-5555  "
    :class="{ 'mobile-toc': isMobile }">

  <div data-label="Table of Contents" class="w-full flex-col items-start md:sticky md:flex md:left-0 ">
    <button class="px-6 gap-2 whitespace-nowrap flex flex-row items-center text-white outline-hidden focus:outline-hidden hover:outline-hidden self-start button-view-outline"
      @click="showToc = !showToc"
      :class="{ 'toc-btn-open': showToc }"
      aria-label="Open Toc">

      <span class="whitespace-nowrap text-[10px] uppercase font-skandia-bold">Outline</span>
      <span class="transform transition-transform duration-150 origin-center text-body-bg">
        <svg width="24" height="13" viewBox="0 0 24 13" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect width="24" height="1.5" fill="#FBFBF6"/>
        <rect y="5.5" width="24" height="1.5" fill="#FBFBF6"/>
        <rect y="11" width="24" height="1.5" fill="#FBFBF6"/>
        </svg>
      </span>
    </button>

    <div x-show="showToc" class="fr-toc bg-black-100 text-body-bg p-6 rounded-[0px_0px_12px_12px] w-[180px] max-h-[300px] lg:max-h-[70vh] overflow-y-auto">
      <div class="js-toc w-full" @click="handleTocClick($event)"></div>
    </div>
  </div>
</div>


    <div id="article-body" data-label="Article Body" class="col-span-12 flex flex-col items-center lg:col-span-8">

      <article class="ghost-content antialiased">

        <p>"I can't tell you how many times we've met with early-stage companies, and they start by telling us their big vision. They say, 'This is what we're about and what we want to change.' But when we ask them what they actually do, they can’t tell us. If you can’t answer that question, don’t do anything else until you can. Nothing else matters."</p><p><strong>This is </strong><strong><a href="https://twitter.com/brooke?ref=review.firstround.com">Brooke Hammerling</a></strong><strong>, unfiltered.</strong> Her company <a href="http://www.brewpr.com/?ref=review.firstround.com">Brew Media</a> <a href="http://www.brewpr.com/?ref=review.firstround.com">Relations </a>match-makes young companies with journalists, influencers and anyone else who can help propel them to stardom. Today, her roster includes <a href="https://wordpress.com/?ref=review.firstround.com">Wordpress</a>, <a href="http://www.charitywater.org/?ref=review.firstround.com">Charity:Water</a>, <a href="https://www.wealthfront.com/?ref=review.firstround.com">Wealthfront</a>, <a href="http://www.oracle.com/index.html?ref=review.firstround.com">Oracle</a> and <a href="https://about.me/?ref=review.firstround.com">About.me</a> — all known for bold, creative communications strategies. She also happened to make the <a href="http://www.nytimes.com/2009/07/05/business/05pr.html?pagewanted=1&_r=3&sq=hammerling&st=cse&scp=1&ref=review.firstround.com">cover of The New York Times’ Sunday business section</a> a few years ago as the poster woman for doing tech PR differently.</p><p>When it comes to whether startups need help in this area, she has a somewhat subversive opinion: they don’t. Even while at Brew, she’s helped hash out PR plans for a number of entrepreneurs who can't afford full-time agencies or in-house support. And she’s got a playbook of tactics for those who want to do it on their own.</p><h2 id="when-to-go-it-alone">When To Go It Alone</h2><p>No matter how many PR agencies or freelance consultants say otherwise, a small startup can pull off a solid media relations strategy without shelling out for help, Hammerling says. In fact, there are only three reasons an early-stage company should consider retaining the services of a firm:</p><ul><li>It’s entering a crowded market. “They need to be able to show why they’re better, why they’re above the fray,” she says.</li><li>It’s a very disruptive company. “If it’s really going to change huge things, like health care, then they need to get out there ahead fast.”</li><li>There’s a legacy CEO involved who has history with the press. “Even if the company isn’t ready for primetime, there will be a lot of attention.”</li></ul><p>If none of these three are true, then relax. "Everybody else should focus that budget on development of the product and building a team internally."</p><h2 id="you-are-your-message-your-message-is-everything">You Are Your Message &amp; Your Message Is Everything</h2><p>When Hammerling takes on a new client, the first thing she does is separate the key members of the team, including the investors. Then she fires questions at them about the product: “What are you? Why are you? Who are you? What problem are you solving and how are you solving it? Why should people care right now?” The idea is to hear what all of them say — where are the differences? Where are the overlaps? What do the people who care most about the company’s success think it is? This is how a narrative is born.</p><p>This was Hammerling’s approach with<a href="https://groupme.com/?ref=review.firstround.com"> GroupMe</a>, the mobile messaging startup bought by Skype in 2011. “It’s a great example because they were entering a very crowded space, but it wasn’t chat and it wasn’t just texting — we didn’t even want to call it an app,” she says. “Instead, we called it a ‘messaging service’ and talked about it in the context of a story: the frustration everyone feels when they can’t communicate with a whole group of friends at a music festival or a party. We were able to really differentiate them as a new way for friends to talk to each other.”</p><p>A startup can use this strategy without a communications team. “You can come up with your own ideas and compare notes, and develop it together. You might end up somewhere you didn’t predict.” She highlights <a href="https://www.uber.com/?ref=review.firstround.com">Uber</a>, not a Brew client, as an expert example of intentional branding. “They offer themselves as a technology company — not a car service. That’s a very specific message that tells you something about who they are and what they do.”</p><blockquote>PR isn’t about hits and it isn’t about placement. It’s about focusing your voice. It’s about finding your place in the market.</blockquote><figure class="kg-card kg-image-card kg-card-hascaption"><img src="https://review.firstround.com/content/images/2056/firstround/cph4fwp9rx2dcphu1qke_brooke_hammerling.png" class="kg-image" alt loading="lazy"><figcaption>Brooke Hammerling, Founder, Brew Media Relations</figcaption></figure><p>The good thing about separating stakeholders is that everyone will give correct answers to the questions being asked — their delivery will just be different. “It’s not like there’s one perfect answer. Everyone will be right. This just gives you the opportunity to say, oh I like how this one person said that, or how so-and-so explained this concept. You can see who phrased things succinctly and who has a better grasp of the longer narrative. Then you can combine the best.”</p><p>The next step is to build what she calls a messaging document, starting with your most succinct, resonant messaging at the top — maybe it’s just one sentence — “It’s what you want to say at cocktail parties,” she says. Below that, you can dive a little deeper with the three key messages you’d want to share with reporters about the specific problem your company is solving. Under that, you can get more detailed. Then make sure everyone has a copy.</p><p>“PR isn’t about hits, it isn’t about placement — it isn’t ‘You pay us and we’ll get you a clip here or a mention on that blog.’ And it isn’t about a first-day bump that gets no traction,” Hammerling says. “It’s about focusing your voice. It’s about finding your place in the market.”</p><p>Developing messaging that resonates can be especially hard for complex technical and enterprise companies. As one of the first PR people working with NetSuite in 2003, Hammerling knows better than most. “We were talking about data systems and software as a service before the cloud was a thing, and no one understood,” she says. “We had to simplify it down and not tell the whole story. NetSuite does a lot of things, but I needed to tell just that one main thing.”</p><p>Some tough love was required. “They had this very very long-winded messaging on what they were doing by explaining the SaaS model of the financial data of ERP and blah blah blah — and I just told them, the mainstream business press doesn’t know this world yet, and they don’t care,” she says. “We filtered it down to this: ‘the future is software as a service: a business that will enable companies to manage the data that is important to them online, and share it across platforms with many people.’ We had to get people to understand how and why it would relate to them. All it took was taking out the jargon and simplifying it so that people could imagine themselves using it.”</p><p>Drawing analogies is a rookie crutch. Saying “It’s like Twitter, only for dog-sitters” can be instructive, but it also takes away from your brand. The better thing to do is to boil your message down to its core, and then layer in other dimensions and functionality little by little. “You may do 25 things and that’s amazing, but what’s that key thing you do? You have to make sure that’s crystal clear. Tell the story around that then layer in the rest later. If you try to pack it all into one announcement or a press release, you’ll lose your audience.”</p><p>Hammerling points to <a href="https://www.wealthfront.com/?ref=review.firstround.com">WealthFront </a>— a Brew client — as one startup doing a good job of turning a complex concept into a single idea. “It’s taking technology, and using it for financial planning — a function that you normally go to humans for. They really had to hone in on a message so that all kinds of people would feel good using it.” You can see it in everything down to their tagline: “We manage your investments for you.” It’s not about technology or business. It’s about the personal ‘we’ taking care of something for ‘you.’</p><p>This is a hard pill for some founders to swallow. Here they’ve spent all this time building a multi-faceted product and they want to talk about all of its capabilities. But this is almost always a mistake, she says. “It’s hard to tell founders this kind of thing — it’s like telling them their child isn’t ready for an honors class yet. They want to fast-forward.” But she’s seen it end badly too many times.</p><p>Hammerling cites photo sharing app Color as a prime — and commonly used — example. It burst onto the scene in 2011, having raised a whopping $41 million in its first round of funding. Its shocking success seemed clear as the company made the rounds with the media — but the app’s actual functionality was not. “Nobody knew what the app did or what to expect once they downloaded it. Because of the money, people were expecting something great, and it just wasn’t there.” Color’s demise was slow and quiet.</p><p>If she had been the one handling the company’s PR, Hammerling says she wouldn’t have let the money dictate the strategy. “Of course they couldn’t keep the funding quiet — too many people were involved and it was an absurd amount. But I wouldn’t have tried to do a full court press around the product at the same time,” she says.</p><h2 id="prepping-for-launch">Prepping for Launch</h2><p>Needless to say, Hammerling has seen a lot of missteps like this and more. She’s had startups come to her three days before their scheduled launch asking for a turnkey media strategy. Of course, there’s no such thing, she says. A solid media plan needs a runway of three to six months. “Even if you have a couple weeks and marketing material, that’s not enough. It’s not going to be effective and it’s going to look fake.” When a company does this — and plenty still do — Hammerling says nine times out of 10 a launch will get botched, and they never get another shot at it.</p><p>Like creating messaging, preparing for an effective launch starts with a list of questions — all designed to suss out your real motivations for doing PR:</p><ul><li>Do you just want a lot of attention early on?</li><li>Is the goal to attract a ton of users? Customers?</li><li>Is the announcement more about recruiting top talent?</li><li>Do you want to raise more capital or VC interest?</li></ul><p>The answers to these questions will shape your approach and your story. For the most part, different goals mean you go after different outlets. If you’re trying to hire great engineers, you want to get on Reddit, Hacker News, or the blogs engineers you want are reading. If you’re trying to get in front of investors, figure out what they read, and pitch those publications. Also, the more transparent you are about your goals, the more likely you are to achieve them, Hammerling says. “Stick with your one basic, overarching message, you just need to tweak it a little bit for each audience.” Sounds simple, but there are some caveats.</p><p><strong>Avoid timing your launch with a funding announcement.</strong> Again, Color is a useful example. “If you need to announce funding, try to separate it as much as you can from your product. Absolutely don’t lead with it.” A large round raises expectations, a small one lowers them. The amount of money can all too easily convince people what they should expect from your company and how likely it is to succeed. The product should have a chance to stand on its own.</p><p>On top of this, a surprising number of companies choose to launch before their product is ready at all. “It’s a head scratcher, but I’ve seen so many startups determined to stick to their timelines that they’ll unveil something before it works. Then all of the bad press and feedback sets them back six months or more,” Hammerling says. “You have to be patient. If you have to pull a plug, pull the marketing plug. Nobody’s setting those deadlines but you.”</p><h2 id="getting-on-the-media%E2%80%99s-radar">Getting on the Media’s Radar</h2><p>Once you have your plan, it’s all about placement. How do you get attention in the deafening echo chamber of today’s technology news cycle? As Hammerling puts it — unsurprisingly — it’s all about relationships. And you don’t need to be a communications pro to make them. The obvious rules apply: be smart, pay attention, and don’t be rude. But there’s a bevy of other hacks for getting noticed.</p><p>“Figure out who’s covering your industry — whether it’s the broader tech industry, internet of things, consumer internet, mobile, whatever,” Hammerling says. “Then there will be those few reporters who cover that specific area you’re in. Make sure you read everything they write. And don’t just look forward, go back in time. Get a sense of their writing style and personality, what topics interest them.” Following reporters on Twitter can also give you a better sense of their life outside of work and what they’re really passionate about. All of this can be used to craft personalized communications.</p><p>While there is no set number of reporters an entrepreneur should work with, there’s value in forging meaningful relationships with a good handful in your area and broader industry, Hammerling says. When she first got her start, the tech media was less built out, and she made it a point getting to know 15 of the most influential reporters in the sector. “Now they’re running their own publications, their own bureaus, their own blogs,” she says. “If you’re a young entrepreneur, get to know the younger guys too — that’s just as important because these are people who share your mindset. They are entrepreneurial too, and they’re going to be the big guys down the road.”</p><p><strong>Don’t be afraid to reach out.</strong>“The reporters covering tech want to hear your story. They’re actively looking to build relationships with entrepreneurs. If you live in the same city, try to set up some time to talk. If you don’t, then drop them a note a few weeks before you pass through their area and say you’d love to grab coffee. Not everyone will say yes, but a lot will,” Hammerling advises. Of course, don’t do this until you have your message down, know it backwards and forwards, and can clearly explain your value proposition to the market. She also advises against talking to reporters directly too far in advance of launch. There’s risk that you’ll tip your hand before your company’s ready. Once your product’s out in the open, that’s the time to focus on maintaining connections.</p><p>The importance of social media is a testament to how much the media relations game has evolved. In the past, it was a common tactic to offer an outlet an exclusive to get their attention — but this isn’t a good hand to play when there are hundreds of influential blogs out there. It’s one thing if you have a massive story that The Wall Street Journal or The New York Times might want to print, but this is hardly the case at the early-stage level.</p><blockquote>I’m against the exclusive around a product launch — I think those days are over.</blockquote><p>That said, exclusives can be useful for damage control. The more bloggers and editors out there, the more people actively uncovering news. It’s not unlikely that a reporter will get ahold of your story or a piece of it before it’s ready for showtime. “In my experience, if a reporter does get your information, you can work with them and build an even better relationship,” Hammerling says. “You can say, ‘You know what, we’re actually not ready for that to go live. We need another week, but we’ll give you everything you need.’ That’s when you give them the exclusive. They uncovered it, they earned it.” In this situation, it’s important to be grateful and show respect, but it’s not something to put into common practice.</p><p>“Giving exclusives will end up hurting your ability to build relationships in the long run,” she explains. “You’re just creating bad blood with all the other journalists who didn’t get the story. In the tech blog world, if someone posted something 30 seconds earlier, it’s perceived as an exclusive, even if it’s the exact same story. So you have to be very careful.”</p><h2 id="once-you-have-their-attention">Once You Have Their Attention</h2><p>If you’ve put in the time cultivating your media network, it’s time to invest even more in learning how to talk to them. And if there’s one person you should pay, it’s a media trainer. “There are experts out there that are exquisitely good at teaching people how to speak to the media,” Hammerling says. “Talking to reporters is a whole different beast, a whole different art entirely. And you can work with a trainer without retaining an agency. You can just hire them to come in and spend half a day session finessing your message and getting you comfortable with answering questions.”</p><p>Hammerling personally recommends<a href="http://www.joedolce.com/?ref=review.firstround.com"> Joe Dolce</a>, a communications expert Brew has been working a lot with as of late. As a former journalist and editor for big titles like Details, and as a public relations executive, he’s been on both sides of interviews and knows what effective answers look like.</p><p>Some founders may need this boost more than others, and it’s important to recognize which type you are. “Technical founders may have a more difficult time because they aren’t comfortable shortening their ideas or statements to be more catchy. They get their point across in minutes when they need to be doing it in seconds,” she says.</p><p>But what if you can’t afford a professional? That doesn’t mean you have to stay quiet. “You can get some of the same lessons from watching videos of successful entrepreneurs who are really really good at telling their stories.” Hammerling lists Salesforce’s <a href="https://www.youtube.com/watch?v=Q8yYWmOz3Ow&feature=youtu.be&ref=review.firstround.com">Marc Benioff</a> and Oracle’s <a href="https://www.youtube.com/watch?v=TAVUNJK7UEA&feature=youtu.be&t=1m&ref=review.firstround.com">Larry Ellison</a> as her top two examples. “They’re just really great at presenting, talking in front of an audience, having a sense of humor. They speak with a twinkle in their eye even when they’re being really serious. Most importantly, they look like they’re having fun. You don’t want to be this glum robotic voice reciting key messages. If you’re passionate and having fun, the media will be enthralled.”</p><p>Another example, a little closer to home for the early-stage crowd, is <a href="https://www.box.com/?ref=review.firstround.com">Box</a> founder and CEO <a href="https://www.youtube.com/watch?v=EvoEbapVoKM&feature=youtu.be&ref=review.firstround.com">Aaron Levie</a>. “Leaders like Aaron are clever and very quick in how they connect to their audience while they’re speaking. You can tell he’s enjoying it, and that he really cares about what he’s saying. It’s rare to see someone his age be this fluid. He doesn’t mince words. He says things that are forward-looking and ahead of the times, and he’s very knowledgeable about the other companies in the space,” she says.</p><p>After all, knowing your competition is vital to being a credible and compelling voice. “When you know your competitive space so well, better than anyone else, you can be a real resource to the media — and the media is constantly looking for good resources. Being able to speak about your company is one thing, but being able to speak about the industry at large makes you multi-dimensional and worth talking to.”</p><blockquote>People have a tendency to think today's news is just tomorrow’s trash, but not now. The internet is forever and people have long memories.</blockquote><p>Keep in mind, being an area expert is different than trash-talking your competitors, and that can be a finer line than many entrepreneurs think. Especially as areas like cloud storage or e-commerce become more crowded, reporters love asking about how companies plan to differentiate themselves or crush their opponents. This is a big red flag. “Negativity never wins,” Hammerling says. “If you’re a huge company, sure you might be able to pull off being snarky or sassy, but as a startup, all you should be is respectful of your competition. If you have to, talk about it like you’re all part of the same community and then bring it back to you.”</p><p>If pressed and you’re forced to cite a competitor, she advises going big. “Don’t talk about the companies you’re asked about specifically, bring up someone who’s related and done it right. Like if you’re in the communications space, talk about<a href="https://www.skype.com/en/?ref=review.firstround.com"> Skype</a> or other big players. Suddenly, you’re connecting your startup with the success of these giants. That’s much better than talking about the various start-ups you’re positioned against — that just pigeonholes you.”</p><p>Remember, every time you interact with the media matters. You need to constantly check yourself to be proactive but not obnoxious, informative but not aggressive. Don’t be that one founder who keeps asking if the story will run — if a reporter says it will, it probably will, Hammerling assures. And never attack someone personally based on what they’ve written on social media or otherwise — even if they’ve gotten a key detail about your company wrong. “Writers take things personally and they remember who they don’t like talking to,” she says. “I remember asking a few reporters what they thought of a particular founder, and they collectively gave the same eye roll. ‘He’s terrible,’ they all said. People have a tendency to think today’s news is just tomorrow’s trash, but not now. The internet is forever and people have long memories.”</p><p>At the same time, being too good at working your relationships, and getting too chummy with the media can have consequences too. “When you think you’re purely friends with a journalist, 90% of the time you’ll end up really unhappy with how a story about you or your company turns out. It’s one thing to be friends, and it’s another thing to work on a story together. You can get blinded and forget to really focus on the specifics and your message. You might give them too much runway or tiptoe around things you shouldn’t. I see it all the time, people with close relationships who are unable to manifest that into great stories because they aren’t thinking with their PR hat on.”</p><p>But this doesn’t mean it’s impossible or ill-advised to forge warm, friendly relationships with media. You just have to make sure they’re clearly defined — and that the terms you use around each other are equally defined. “Some people say you can’t genuinely be friends with reporters, but I think that’s garbage. You just have to be clear. I always tell entrepreneurs know what off the record and what on background mean — and know when to say them. If you’re talking to someone, you can’t give them all this intelligence and then afterwards say, ‘Oh by the way, that was all off the record.’ That’s not how it works.”</p><p>Another thing to realize is that with comment threads and social media, readers have become a concrete part of the media experience. “Younger CEOs take comments less seriously than older ones, and that’s a good thing. The only way to handle it is to learn. You can’t make it personal and go off on someone, because that’ll just blow up into something bigger,” Hammerling says. She’s seen her share of CEOs kept up at night reading all the comments on every blog post about their product. It’s hard to dissuade them, but at the very least she encourages them to identify takeaways.</p><p>“You may see someone commenting on how they love an aspect of your product. Then you know that’s something that’s really resonating that you can start honing your message around. Inversely, if people are responding negatively to something, then you can very dynamically change your messaging around that, make it more clear or consider changing that aspect of the product.”</p><h2 id="the-irreplaceable-voice-of-the-founder">The Irreplaceable Voice of the Founder</h2><blockquote>The vast majority of the time, the founders are the voice and heart of the company.</blockquote><p>It’s telling that founders are the ones up at night, obsessively reading comments, thinking about how they could do better next time. They care the most about the company’s success — and that devotion is like energy in a bottle. “It’s a particular type of energy that inspires people, makes for good stories, and can’t be substituted.” Hammerling believes this so deeply that she refuses to work with clients if a founder is not willing to be actively involved in media relations.</p><p>This is why, when it comes to announcements, there’s an increasingly strong preference toward a founder’s blog post over an impersonal press release. While you might see some PR teams or companies compiling literature for media — everything from headshots to packets of pre-vetted executive quotes and market position data — this comes off as disingenuous more than it helps.</p><p>“When a founder decides to take a backseat, and put a marketing person out there, it negatively impacts the team, the organization, and the trust reporters have in them,” she says. The press really wants that connection with the founder and their story.” This should never change, no matter what stage you’re at, even if you have an agency or a big in-house comms team.</p><p>In fact, there’s no better argument for early-stage startups’ ability to navigate the media field on their own — at least at first. When a product, message and strategy align with a founder who can deliver it all clearly and persuasively, there’s no telling how powerful that can be.</p><p>As Hammerling puts it, an agency putting out a press release says “agenda,” a passionate founder going straight to the press says, “Hey, look at this, it’s going to be cool.”</p>
      </article>

      <div class="flex flex-row justify-center items-center border-t border-black/20 pt-8 pb-8 max-w-content m-auto w-full max-lg:px-5 gap-x-4">
        <span class="text-sm text-black">Share on:</span>
          <div class="flex flex-row justify-center md:flex xl:col-span-8" data-share>
    <div data-label="Share Article" class="flex flex-row space-y-0 sm:space-y-12 items-center">
      <div class="flex flex-row items-center space-x-2 sm:space-x-2">
        <div data-label="Share Options" class="flex flex-row space-x-2 sm:space-x-2">
          <a class="flex flex-row justify-center items-center p-3 ibtn transition duration-200 rounded-full outline-hidden focus:outline-hidden hover:outline-hidden active:outline-hidden border border-black/25 h-[40px] w-[40px] text-black"  target="_blank"
            href="https://twitter.com/share?text=Why%20Most%20Startups%20Don't%20'Get'%20Press&amp;url=https://review.firstround.com/why-most-startups-dont-get-press/"
            title="Share on Twitter" aria-label="Tweet">
          <i class="icon icon-twitter-x stroke-1.5">
  <svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-brand-x hover:contrast-90 hover:brightness-110" width="16" height="16" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
  <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
  <path d="M4 4l11.733 16h4.267l-11.733 -16z"></path>
  <path d="M4 20l6.768 -6.768m2.46 -2.46l6.772 -6.772"></path>
</svg>
</i>          </a>
          <a class="flex flex-row justify-center items-center p-3 ibtn rounded-full transition duration-200 outline-hidden focus:outline-hidden hover:outline-hidden active:outline-hidden border border-black/25 h-[40px] w-[40px] text-black"target="_blank"
            href="https://www.linkedin.com/shareArticle?mini=true&url=https://review.firstround.com/why-most-startups-dont-get-press/&title=Why%20Most%20Startups%20Don't%20'Get'%20Press&summary=Why%20Most%20Startups%20Don't%20'Get'%20Press"
            title="Share on Linkedin" aria-label="Share">
            <i class="icon icon-linkedin stroke-1.5">
  <svg width="14" height="13" viewBox="0 0 14 13" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M9.59995 4.28955C10.6343 4.28955 11.6263 4.6888 12.3577 5.39946C13.0891 6.11013 13.5 7.07399 13.5 8.07902V12.5001H10.9V8.07902C10.9 7.74401 10.763 7.42273 10.5192 7.18584C10.2754 6.94895 9.94473 6.81587 9.59995 6.81587C9.25517 6.81587 8.92451 6.94895 8.68071 7.18584C8.43692 7.42273 8.29995 7.74401 8.29995 8.07902V12.5001H5.69995V8.07902C5.69995 7.07399 6.11084 6.11013 6.84223 5.39946C7.57363 4.6888 8.56561 4.28955 9.59995 4.28955Z" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M3.1 4.9209H0.5V12.4998H3.1V4.9209Z" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M1.8 3.02632C2.51797 3.02632 3.1 2.46078 3.1 1.76316C3.1 1.06554 2.51797 0.5 1.8 0.5C1.08203 0.5 0.5 1.06554 0.5 1.76316C0.5 2.46078 1.08203 3.02632 1.8 3.02632Z" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/>
</svg>

</i>          </a>
        </div>
      </div>
    </div>
  </div>
      </div>
    </div>
  </div>

  

<div class="ctx-module-container ctx_shortcode_placement ctx-clearfix"></div>


    </main>

    <footer data-label="Site Footer" class="footer relative w-full flex-col bg-black text-body-bg px-5 py-14 lg:px-0 ">


    <div class="max-w-7xl m-auto w-full lg:px-5">

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">

      <div data-label="Feed Links" class="flex space-y-4 flex-col grow text-xl font-skandia-medium tracking-tight max-w-[420px]">


        <nav>

            <ul class="grid grid-cols-2 gap-4" data-nav-items>
      <li class="nav-about  nav-item"
          data-label="About" data-length="5" data-slug="about">
        <a class="" href="https://review.firstround.com/about/">
          <span>About</span>
        </a>
        <div id="tag-content-about" class="dropdown-content"></div>
      </li>
      <li class="nav-articles  nav-item"
          data-label="Articles" data-length="8" data-slug="articles">
        <a class="" href="https://review.firstround.com/articles/">
          <span>Articles</span>
        </a>
        <div id="tag-content-articles" class="dropdown-content"></div>
      </li>
      <li class="nav-curated-reads  nav-item"
          data-label="Curated Reads" data-length="13" data-slug="curated-reads">
        <a class="" href="https://review.firstround.com/collections/">
          <span>Curated Reads</span>
        </a>
        <div id="tag-content-curated-reads" class="dropdown-content"></div>
      </li>
      <li class="nav-paths-to-pmf  nav-item"
          data-label="Paths to PMF" data-length="12" data-slug="paths-to-pmf">
        <a class="" href="https://review.firstround.com/series/product-market-fit/">
          <span>Paths to PMF</span>
        </a>
        <div id="tag-content-paths-to-pmf" class="dropdown-content"></div>
      </li>
      <li class="nav-newsletter  nav-item"
          data-label="Newsletter" data-length="10" data-slug="newsletter">
        <a class="" href="https://review.firstround.com/newsletter/">
          <span>Newsletter</span>
        </a>
        <div id="tag-content-newsletter" class="dropdown-content"></div>
      </li>
      <li class="nav-podcast  nav-item"
          data-label="Podcast" data-length="7" data-slug="podcast">
        <a class="" href="https://review.firstround.com/podcast/">
          <span>Podcast</span>
        </a>
        <div id="tag-content-podcast" class="dropdown-content"></div>
      </li>
      <li class="nav-sign-in  nav-item"
          data-label="Sign in" data-length="7" data-slug="sign-in">
        <a class="" href="https://review.firstround.com/#/portal/signin">
          <span>Sign in</span>
        </a>
        <div id="tag-content-sign-in" class="dropdown-content"></div>
      </li>
      <li class="nav-glossary  nav-item"
          data-label="Glossary" data-length="8" data-slug="glossary">
        <a class="" href="https://review.firstround.com/glossary/">
          <span>Glossary</span>
        </a>
        <div id="tag-content-glossary" class="dropdown-content"></div>
      </li>

     <li class="nav- nav-current  px-4 flex items-center max-lg:py-6 max-lg:px-0 max-lg:text-left fr-signin"
      data-label="" data-length="">
      <a href="/signin/" class="text-sm lg:px-2 leading-none h-[30px] flex items-center justify-center max-lg:font-skandia-bold max-lg:text-3xl max-lg:leading-[44.8px] max-lg:tracking-[-0.03em] max-lg:cursor-pointer max-lg:text-white" data-portal="signin"><span>Sign In</span></a>
    </li>
</ul>

<style>
  /* Dropdown parent - maintain existing layout */
  .dropdown-parent {
    position: relative;
  }

  /* Dropdown arrow indicator */
  .dropdown-parent .dropbtn {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
  }

  .dropdown-parent .dropdown-arrow {
    opacity: 0.6;
    transition: transform 0.3s ease, opacity 0.3s ease;
    margin-left: 0.25rem;
    flex-shrink: 0;
  }

  /* Dropdown content - positioned absolutely, doesn't affect layout */
  .dropdown-content {
    display: none;
    position: absolute;
    top: 100%;
    left: 0;
    background-color: var(--color-body-bg, #fff);
    min-width: 200px;
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    z-index: 1000;
    padding: 0.5rem 6px;
    margin-top: 0.5rem;
    border-radius: 0.25rem;
    border: 1px solid rgba(0, 0, 0, 0.05);
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.3s ease, visibility 0.3s ease, transform 0.3s ease;
    transform: translateY(-10px);
  }

  /* Desktop: Show dropdown on hover */
  @media (min-width: 1024px) {
    .dropdown-parent:hover .dropdown-arrow {
      opacity: 1;
      transform: rotate(180deg);
    }

    .dropdown-parent:hover .dropdown-content {
      display: block;
      opacity: 1;
      visibility: visible;
      transform: translateY(0);
      top: 48px;
    }
  }

  .dropdown-content.show {
    display: block;
    opacity: 1;
    visibility: visible;
    transform: translateY(0);
  }

  /* Dropdown items - match theme styling */
  .dropdown-content .dropdown-item {
    display: block;
    transition: opacity 0.3s ease;
    padding: 0 6px;
  }

  .dropdown-content .dropdown-item a {
    display: block;
    padding: 0.625rem 0.625rem;
    text-decoration: none;
    color: var(--color-typography, #000);
    font-family: var(--font-body);
    font-size: 0.875rem;
    transition: all 0.3s ease;
  }

  .dropdown-content .dropdown-item a:hover {
    background-color: rgba(0, 0, 0, 0.03);
    font-weight: 500;
  }

  /* Mobile: dropdown items should look like regular menu items */
  @media (max-width: 1023px) {
    .dropdown-content .dropdown-item {
      padding: 0 !important;
      background-color: transparent !important;
    }

    .dropdown-content .dropdown-item a {
      background-color: transparent !important;
      padding: 0.5rem 0 !important;
    }

    .dropdown-content .dropdown-item a:hover {
      background-color: transparent !important;
    }
  }

  /* Mobile: Disable hover completely, only allow click on arrow */
  @media (max-width: 1023px) {
    /* Disable ALL hover effects on mobile - prevent any hover state */
    .dropdown-parent:hover .dropdown-content.show {
      display: block !important;
      opacity: 1 !important;
      visibility: visible !important;
      transform: none !important;
      transition: none !important;
      pointer-events: auto !important;
    }

    .dropdown-parent:hover .dropdown-arrow {
      transform: rotate(-90deg) !important;
      opacity: 0.8 !important;
      transition: none !important;
    }
  }

  /* Touch devices: Completely disable hover (more specific than max-width) */
  @media (hover: none) and (pointer: coarse) {
    .dropdown-parent:hover .dropdown-content {
      display: none !important;
      opacity: 0 !important;
      visibility: hidden !important;
      transform: none !important;
      transition: none !important;
      pointer-events: none !important;
    }

    .dropdown-parent:hover .dropdown-arrow {
      transform: rotate(-90deg) !important;
      opacity: 0.8 !important;
      transition: none !important;
    }
  }

  @media (max-width: 1023px) {
    .dropdown-content {
      position: static !important;
      background-color: transparent !important;
      box-shadow: none !important;
      border: none !important;
      margin-top: 0 !important;
      padding-left: 1rem;
      padding-right: 0 !important;
      transform: none !important;
      transition: none !important;
      display: none !important;
      opacity: 1 !important;
      visibility: hidden !important;
      min-width: auto !important;
      pointer-events: none !important;
    }

    /* Only show when .show class is added via JS click */
    .dropdown-content.show {
      display: block !important;
      opacity: 1 !important;
      visibility: visible !important;
      pointer-events: auto !important;
    }

    .dropdown-parent .dropdown-arrow {
      transform: rotate(-90deg);
      cursor: pointer;
      pointer-events: auto !important;
      opacity: 0.8;
      min-width: 32px;
      min-height: 32px;
      padding: 8px;
      margin-left: 0.5rem;
      display: inline-flex !important;
      align-items: center;
      justify-content: center;
      -webkit-tap-highlight-color: transparent;
      touch-action: manipulation;
    }

    .dropdown-parent .dropdown-arrow:active {
      opacity: 1;
    }

    .dropdown-parent .dropdown-arrow.show {
      transform: rotate(90deg);
      opacity: 1;
    }

    /* Mobile menu specific styling */
    [data-label="Site Menu"] .dropdown-parent .dropbtn {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
    }

    [data-label="Site Menu"] .dropdown-content {
      position: static !important;
      background-color: transparent !important;
      box-shadow: none !important;
      border: none !important;
      padding: 0 !important;
      margin-top: 0 !important;
      margin-left: 0 !important;
      min-width: auto !important;
      opacity: 1 !important;
      visibility: visible !important;
      transform: none !important;
    }

    [data-label="Site Menu"] .dropdown-content .dropdown-item {
      padding: 0 !important;
      display: block;
    }

    [data-label="Site Menu"] .dropdown-content .dropdown-item a {
      font-size: 1.5rem;
      line-height: 2rem;
      padding: 0.5rem 0;
      color: rgba(255, 255, 255, 0.8);
      display: block;
    }

    [data-label="Site Menu"] .dropdown-content .dropdown-item a:hover {
      color: white;
      background-color: transparent;
    }
  }
</style>

<script>
  function initDropdowns() {
    // Organize sub-items into their parent dropdowns
    // Handle all navigation instances (desktop and mobile menu)
    const navContainers = document.querySelectorAll('[data-nav-items]');
    if (!navContainers || navContainers.length === 0) return;
    
    navContainers.forEach(function(navContainer) {
      const navItems = Array.from(navContainer.querySelectorAll('li'));
      const subItems = Array.from(navContainer.querySelectorAll('.dropdown-subitem'));
      
      // First, identify which nav items should have dropdowns (those with sub-items following them)
      const itemsWithDropdowns = new Set();
      
      subItems.forEach(function(subItem) {
        let prevSibling = subItem.previousElementSibling;
        // Look backwards to find the parent nav item
        while (prevSibling) {
          if (prevSibling.classList.contains('nav-item') && !prevSibling.classList.contains('dropdown-subitem')) {
            itemsWithDropdowns.add(prevSibling);
            break;
          }
          prevSibling = prevSibling.previousElementSibling;
        }
      });
      
      // Add dropdown functionality only to items that have sub-items
      itemsWithDropdowns.forEach(function(parentItem) {
        parentItem.classList.add('dropdown-parent');
        const link = parentItem.querySelector('a');
        if (link) {
          link.classList.add('dropbtn');
          // Add arrow SVG
          const arrow = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
          arrow.setAttribute('class', 'dropdown-arrow');
          arrow.setAttribute('width', '12');
          arrow.setAttribute('height', '12');
          arrow.setAttribute('viewBox', '0 0 12 12');
          arrow.setAttribute('fill', 'none');
          arrow.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
          const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
          path.setAttribute('d', 'M3 4.5L6 7.5L9 4.5');
          path.setAttribute('stroke', 'currentColor');
          path.setAttribute('stroke-width', '1.5');
          path.setAttribute('stroke-linecap', 'round');
          path.setAttribute('stroke-linejoin', 'round');
          arrow.appendChild(path);
          link.appendChild(arrow);
        }
      });
      
      // Process sub-items in reverse order to avoid index issues
      subItems.reverse().forEach(function(subItem) {
        if (!subItem.parentNode) return; // Already moved
        
        // Find the parent item that immediately precedes this sub-item
        let currentParent = null;
        let prevSibling = subItem.previousElementSibling;
        
        // Look backwards through siblings to find the most recent parent item
        while (prevSibling) {
          if (prevSibling.classList.contains('dropdown-parent')) {
            currentParent = prevSibling;
            break;
          }
          prevSibling = prevSibling.previousElementSibling;
        }
        
        if (currentParent) {
          const dropdown = currentParent.querySelector('.dropdown-content');
          if (dropdown) {
            // Clone and move the sub-item into the dropdown
            const clonedItem = subItem.cloneNode(true);
            clonedItem.classList.remove('dropdown-subitem');
            clonedItem.classList.add('dropdown-item');
            // Remove onclick from cloned link
            const clonedLink = clonedItem.querySelector('a');
            if (clonedLink) {
              clonedLink.removeAttribute('onclick');
              // Clean the label text (remove "- " prefix)
              const labelSpan = clonedLink.querySelector('span');
              if (labelSpan && labelSpan.textContent.trim().startsWith('-')) {
                labelSpan.textContent = labelSpan.textContent.trim().substring(1).trim();
              }
            }
            dropdown.appendChild(clonedItem);
            subItem.remove();
          }
        }
      });
    }); // End forEach navContainer
  }

  // Run immediately if DOM is ready, otherwise wait
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDropdowns);
  } else {
    initDropdowns();
  }

  // Re-initialize when mobile menu opens (in case it's rendered dynamically)
  document.addEventListener('DOMContentLoaded', function() {
    // Watch for mobile menu opening
    const mobileMenuTrigger = document.querySelector('[data-label="Site Menu"]');
    if (mobileMenuTrigger) {
      // Use MutationObserver to detect when mobile menu content is added
      const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
          if (mutation.addedNodes.length > 0) {
            // Check if navigation items were added
            const hasNavItems = Array.from(mutation.addedNodes).some(function(node) {
              return node.nodeType === 1 && (
                node.querySelector('[data-nav-items]') || 
                node.classList?.contains('dropdown-subitem') ||
                node.querySelector?.('.dropdown-subitem')
              );
            });
            if (hasNavItems) {
              setTimeout(initDropdowns, 50);
            }
          }
        });
      });
      
      observer.observe(mobileMenuTrigger, {
        childList: true,
        subtree: true
      });
    }
    
    // Also re-initialize on any click that might open mobile menu
    document.addEventListener('click', function(e) {
      if (e.target.closest('[data-label="Site Menu"]') || e.target.closest('.mobile-menu')) {
        setTimeout(initDropdowns, 100);
      }
    });
  });

  // Add mobile click support for dropdown arrows
  function setupMobileDropdowns() {
    const arrows = document.querySelectorAll('.dropdown-arrow');
    
    arrows.forEach(function(arrow) {
      // Skip if already has click handler
      if (arrow.dataset.clickSetup === 'true') return;
      arrow.dataset.clickSetup = 'true';
      
      const parent = arrow.closest('.dropdown-parent');
      if (!parent) return;
      
      const dropdown = parent.querySelector('.dropdown-content');
      if (!dropdown) return;
      
      // Make arrow clickable
      arrow.style.pointerEvents = 'auto';
      arrow.style.cursor = 'pointer';
      
      // Arrow click handler - toggle dropdown on mobile
      arrow.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        
        // Only work on mobile (max-width: 1023px)
        if (window.innerWidth > 1023) return;
        
        const isShowing = dropdown.classList.contains('show');
        
        // Close all other dropdowns first
        document.querySelectorAll('.dropdown-content.show').forEach(function(dd) {
          if (dd !== dropdown) {
            dd.classList.remove('show');
          }
        });
        document.querySelectorAll('.dropdown-arrow.show').forEach(function(arr) {
          if (arr !== arrow) {
            arr.classList.remove('show');
          }
        });
        
        // Toggle current dropdown
        if (isShowing) {
          dropdown.classList.remove('show');
          arrow.classList.remove('show');
        } else {
          dropdown.classList.add('show');
          arrow.classList.add('show');
        }
      });
    });
    
    // Close dropdowns when clicking outside (mobile only)
    if (!document.body.dataset.dropdownOutsideClick) {
      document.body.dataset.dropdownOutsideClick = 'true';
      document.addEventListener('click', function(event) {
        if (window.innerWidth > 1023) return;
        
        // If click is not on arrow or dropdown, close all
        if (!event.target.closest('.dropdown-arrow') && !event.target.closest('.dropdown-content')) {
          document.querySelectorAll('.dropdown-content.show').forEach(function(dd) {
            dd.classList.remove('show');
          });
          document.querySelectorAll('.dropdown-arrow.show').forEach(function(arr) {
            arr.classList.remove('show');
          });
        }
      });
    }
  }

  // Setup mobile dropdowns after initialization
  function initializeMobileDropdowns() {
    // Wait for dropdowns to be initialized by initDropdowns()
    setTimeout(setupMobileDropdowns, 150);
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeMobileDropdowns);
  } else {
    initializeMobileDropdowns();
  }
  
  // Re-run setup when mobile menu opens (dynamic content)
  document.addEventListener('click', function(e) {
    if (e.target.closest('[data-label="Site Menu"]') || e.target.closest('.mobile-menu')) {
      setTimeout(setupMobileDropdowns, 150);
    }
  });
</script>

          </nav>


      </div>

      <div class="border border-white/25 rounded-sm p-8 founders-notepad">

        <span class="text-2xl !font-skandia-medium text-body-bg mb-4 block">For the founder's notepad:</span>


          <div class="blockquotes-content font-leif text-xl tracking-tight">
              <blockquote>"If you personally want to grow as fast as your company, you have to give away your job every couple months." – <a href="https://review.firstround.com/give-away-your-legos-and-other-commandments-for-scaling-startups/"><strong><u>Molly Graham</u></strong></a> </blockquote><blockquote>“Asking ‘Why can't this be done sooner?’ methodically, reliably and habitually can have a profound impact on the speed of your organization.” – <a href="https://review.firstround.com/speed-as-a-habit/"><u>Dave Girouard</u></a></blockquote><blockquote>“End every meeting or conversation with the feeling and optimism you’d like to have at the start of your next conversation with the person.” – <a href="https://review.firstround.com/how-to-become-insanely-well-connected/"><u>Chris Fralic</u></a></blockquote><blockquote>“Focus is doing things with a clear intention. It doesn’t mean you charge single-minded toward a goal. It means you pay rapt and incremental attention to how you need to turn the rudder on a project.” – <a href="https://review.firstround.com/how-facebooks-vp-of-product-finds-focus-and-creates-conditions-for-intentional-work/"><u>Fidji Simo</u></a></blockquote><blockquote>“It’s essential to grow with the company — rather than having the company grow around you.” – <a href="https://review.firstround.com/23-tactical-company-building-lessons-learned-from-scaling-stripe-and-notion/"><u>Cristina Cordova</u></a>&nbsp;</blockquote><blockquote>“You have to be impatient with shipping, but patient with your career.” – <a href="https://review.firstround.com/how-to-scale-yourself-down-not-up-as-a-leader/"><u>James Everingham</u></a></blockquote><blockquote>“‘I trust you, make the call’ might be the six most powerful words you can hear from a manager.”  – <a href="https://review.firstround.com/the-25-micro-habits-of-high-impact-managers"><u>Sean Twersky</u></a></blockquote><blockquote>“Your job as a CEO is to build fire departments, not put out fires.” – <a href="https://review.firstround.com/an-exact-breakdown-of-how-one-ceo-spent-his-first-two-years-of-company-building/"><u>Sam Corcos</u></a>&nbsp;</blockquote><blockquote>“Can you say with confidence that each report would want to be on your team again? If you aren’t sure that the answer is yes, it’s probably no — much like how if you have to ask, ‘Am I in love?’ you’re probably not.” – <a href="https://review.firstround.com/the-essential-questions-that-have-powered-this-top-silicon-valley-managers-career/"><u>Julie Zhuo</u></a>&nbsp;</blockquote><blockquote>“People can get addicted to yak shaving. An effective engineering generalist knows when to move on. Pay attention to whether they used their time wisely, not just the results.” – <a href="https://review.firstround.com/how-instagram-co-founder-mike-krieger-took-its-engineering-org-from-0-to-300-people/"><u>Mike Krieger</u></a>&nbsp;</blockquote><blockquote>“It sounds so simple to say that bosses need to tell employees when they're screwing up. But it very rarely happens.” – <a href="https://review.firstround.com/radical-candor-the-surprising-secret-to-being-a-good-boss/"><u>Kim Scott</u></a></blockquote><blockquote>“You’ll know you understand the customer problem enough when you can predict 75% of what a customer tells you. Keep having these conversations until three-quarters of it is stuff you already know.” – <a href="https://review.firstround.com/vantas-path-to-product-market-fit/"><u>Christina Cacioppo</u></a></blockquote><blockquote>“I have a rule: no company swag until the business has at least $250K of revenue or 250k users. Until then, you don’t get to “feel” the benefits of having started a company.” – <a href="https://review.firstround.com/the-minimum-viable-testing-process-for-evaluating-startup-ideas/"><u>Gagan Biyani</u></a></blockquote><blockquote>“The business model ends up becoming the business. It’s equally important as the market you’re going after and the product that you build.” – <a href="https://review.firstround.com/unpacking-5-of-atlassians-most-unconventional-company-building-moves/"><u>Jay Simons</u></a>&nbsp;</blockquote><blockquote>“If speed is the yin, the yang is prioritization. You can’t be fast if you don’t know what’s important.” – <a href="https://review.firstround.com/your-marketing-org-is-slow-heres-a-framework-to-move-faster/"><u>Jaleh Rezaei</u></a></blockquote><blockquote>“If you treat your connections as a kind of personal ATM you use for frequent withdrawals, you’ll quickly be disappointed (and overdrawn).”  – <a href="https://review.firstround.com/how-to-make-connections-that-count-advice-from-a-silicon-valley-veteran-and-introvert/"><u>Karen Wickre</u></a>&nbsp;</blockquote><blockquote>“Delighting the customer always yields better returns than countering or copying a competitor. It’s just a lot harder to do.” – <a href="https://review.firstround.com/When-it-Comes-to-Market-Leadership-Be-the-Gorilla"><u>Andy Rachleff</u></a>&nbsp;</blockquote><blockquote>“When you’re a founder, every moment you’re not writing code or getting users, you need to be making a conscious choice: Is whatever you’re doing worth your time?” – <a href="https://review.firstround.com/the-donts-that-turned-alexis-ohanian-into-the-leader-he-is-today/"><u>Alexis Ohanian</u></a></blockquote><blockquote>“‘Why would a customer not want this?’ is often a far more interesting question than why they would.” – <a href="https://review.firstround.com/ask-why-it-wont-work-and-other-lessons-this-founder-relies-on-while-building-from-0-to-1/"><u>Rick Song</u></a></blockquote><blockquote>“When you leave the planning process wondering if you put too many resources behind a single bet, that’s the bet that ends up succeeding. Bold ideas need bold resourcing.” – <a href="https://review.firstround.com/the-secret-to-a-great-planning-process-lessons-from-airbnb-and-eventbrite/"><u>Lenny Rachitsky and Nels Gilbreth</u></a></blockquote><blockquote>“Treat customer development as a one-on-one with a direct report — you just want to ask the hard questions.” – <a href="https://review.firstround.com/8-product-hurdles-every-founder-must-clear-this-pm-turned-founder-shares-his-playbooks/"><u>Ryan Glasgow</u></a></blockquote>
          </div>


        <a href="javasript:" class="founders-shuffle inline-flex items-center line-link line-link-white gap-2 text-sm mt-4"><span>Shuffle</span><i class="icon icon-arrow-r w-4 h-4">
  <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M0.285645 7.99983L15.5714 7.99983M15.5714 7.99983L9.28564 1.71411M15.5714 7.99983L9.28564 14.2855" stroke="currentColor"/>
</svg>

</i></a>

      </div>

      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-8 border-t border-t-white/20 pt-8 mt-18">

      <div data-label="Branding" class="w-full flex justify-start gap-4 items-center">
          <p class="text-sm">Published by</p><svg width="80" height="43" viewBox="0 0 80 43" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M0 0V19.1483H4.46832V11.4445H10.3048L11.1137 7.86618H4.46832V3.57977H11.9616L12.7705 0H0Z" fill="#FBFBF6"/>
<path d="M31.8959 5.63972L31.063 9.54876C30.8134 9.40894 30.23 9.21499 29.4527 9.21499C28.4529 9.21499 27.7328 9.77127 27.399 10.2148C27.0652 10.6583 26.7044 11.5469 26.7044 13.0729V19.15H22.2646V5.46682H26.7044L26.6758 7.40931H26.7315C27.5088 5.91185 28.6469 5.21875 30.1173 5.21875C31.1622 5.21875 31.8929 5.63972 31.8929 5.63972" fill="#FBFBF6"/>
<path d="M42.1911 18.2612C43.1909 17.4568 43.8013 16.3472 43.8013 14.931C43.8013 13.5147 43.1353 12.6006 42.6902 12.1555C42.2181 11.7105 41.525 11.3512 40.9146 11.1287C40.3313 10.9062 39.9148 10.7663 38.9722 10.4897C38.028 10.2131 36.974 9.87928 36.974 9.15912C36.974 8.43895 37.5574 8.15931 38.6113 8.15931C39.7209 8.15931 41.5762 9.04936 42.4151 9.76201L43.1037 6.58366C42.2257 5.96122 40.6936 5.18994 38.2505 5.18994C35.3909 5.18994 33.0605 6.82723 33.0605 9.49289C33.0605 11.936 34.7534 12.796 35.3909 13.0727C36.0299 13.3493 36.8056 13.5733 37.3334 13.7116C37.8325 13.8785 39.8306 14.3221 39.8306 15.2106C39.8306 16.0992 39.0533 16.3758 37.9438 16.3758C36.8342 16.3758 35.0015 15.6541 33.8919 14.7099L32.2261 17.347C33.7807 18.8189 35.9442 19.4564 37.9994 19.4564C39.8306 19.4564 41.1913 19.04 42.1896 18.2627" fill="#FBFBF6"/>
<path d="M54.199 8.85225H51.5168V13.7641C51.5168 14.9293 51.6821 15.2074 51.9047 15.457C52.0986 15.6795 52.3752 15.7893 52.9586 15.7893C53.6532 15.7893 54.2907 15.5397 54.2907 15.5397L54.5132 19.0097C54.1809 19.0924 52.9586 19.4533 51.3499 19.4533C49.7412 19.4533 48.7143 18.9827 48.0197 18.1753C47.3266 17.3709 47.0484 16.2328 47.0484 14.152V8.85075H45.106V5.49198H47.0484V1.77539H51.5168V5.49499H54.9101L54.199 8.85375V8.85225Z" fill="#FBFBF6"/>
<path d="M10.2958 34.5266C11.7105 34.0831 14.0424 32.6668 14.0424 29.0314C14.0424 27.1175 13.2095 25.59 11.9331 24.6458C10.6837 23.7302 9.01782 23.2852 6.88139 23.2852H0V42.4364H4.49538V35.1085H5.91165L9.93644 42.4364H15.1535L10.2973 34.5251L10.2958 34.5266ZM8.54874 31.1423C7.99396 31.5588 7.27229 31.67 6.24542 31.67H4.46982V26.7582H6.24542C7.27229 26.7582 7.99396 26.8409 8.54874 27.2588C9.13058 27.6753 9.51998 28.2301 9.51998 29.2013C9.51998 30.1726 9.13058 30.7259 8.54874 31.1438" fill="#FBFBF6"/>
<path d="M23.1046 28.5049C18.7746 28.5049 15.5 31.2247 15.5 35.6359C15.5 40.0471 18.7746 42.7413 23.1046 42.7413C27.4346 42.7413 30.6821 40.0501 30.6821 35.6359C30.6821 31.2217 27.4346 28.5049 23.1046 28.5049ZM23.1046 39.1059C21.1335 39.1059 19.9969 37.5799 19.9969 35.5532C19.9969 33.5265 21.135 32.0005 23.1046 32.0005C25.0741 32.0005 26.2122 33.528 26.2122 35.5532C26.2122 37.5784 25.047 39.1059 23.1046 39.1059Z" fill="#FBFBF6"/>
<path d="M46.0811 42.437H41.8068L41.7797 40.8824C41.1963 41.7154 39.753 42.7407 37.7835 42.7407C35.8139 42.7407 34.5645 42.0476 33.8699 41.132C33.2039 40.1878 32.8701 38.9941 32.8701 37.0245V28.7539H37.3099V36.4141C37.3099 37.9943 37.7549 39.0783 39.3366 39.0783C40.9182 39.0783 41.6414 37.8289 41.6414 36.0247V28.7539H46.0811V42.437Z" fill="#FBFBF6"/>
<path d="M62.5604 42.4342H58.1206V35.3002C58.1206 33.496 57.8425 32.1369 56.0113 32.1369C54.18 32.1369 53.5696 33.6915 53.5696 35.6626V42.4342H49.1299V28.7511H53.5696L53.5411 30.7206H53.5967C54.3454 29.5013 55.5948 28.5 57.5659 28.5C61.535 28.5 62.5604 31.0544 62.5604 34.3575V42.4327V42.4342Z" fill="#FBFBF6"/>
<path d="M75.5611 23.2861V30.4186C75.1161 29.7511 73.7299 28.5032 71.2311 28.5032C67.2619 28.5032 64.7646 31.5838 64.7646 35.6086C64.7646 39.6334 67.2619 42.7396 71.2311 42.7396C73.0623 42.7396 74.6996 42.0465 75.7265 40.6588L75.8092 40.6317L75.7806 42.4359H79.9978V23.2861H75.5581H75.5611ZM72.5075 39.1057C70.5936 39.1057 69.2886 37.6624 69.2886 35.553C69.2886 33.4436 70.5936 32.0003 72.5075 32.0003C74.4214 32.0003 75.6994 33.4436 75.6994 35.553C75.6994 37.6624 74.423 39.1057 72.5075 39.1057Z" fill="#FBFBF6"/>
<path d="M14.1927 5.46777L13.4185 9.04604H15.2918V19.1494H19.7315V5.46777H14.1927Z" fill="#FBFBF6"/>
</svg>

      </div>

      <div data-label="Account Links" class="flex flex-col grow space-y-4 justify-end items-start md:items-end">
          <nav class="flex md:space-x-4 items-start md:items-center flex-col-reverse md:flex-row gap-5 md:gap-0">
            
            <div class="flex md:space-x-4 items-center flex-row md:flex-row gap-4 md:gap-0">
              <a class="md:text-sm text-xs hover:underline" target="_blank" href="https://www.firstround.com/privacy" title="Privacy Policy">
              Privacy Policy
              </a>

              <a class="md:text-sm text-xs hover:underline" target="_blank" href="https://www.firstround.com/terms" title="Terms of Service">
              Terms of Service
              </a>
            </div>

            <div class="flex md:space-x-4 items-center justify-center flex-row md:flex-row gap-4 md:gap-0">

            <a class="rounded-full border border-white/20 w-10 h-10 flex items-center justify-center" target="_blank" href="https://twitter.com/firstround">
                <i class="icon icon-twitter fill-current visible w-4 h-4 text-body-bg">
  <svg width="16" height="17" viewBox="0 0 16 17" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M2.0292 2.37061L6.66227 8.98929L2 14.3706H3.04937L7.13127 9.65908L10.4292 14.3706H14L9.10617 7.37969L13.4458 2.37061H12.3965L8.63737 6.70969L5.60001 2.37061H2.0292ZM3.57235 3.19638H5.21276L12.4567 13.5448H10.8163L3.57235 3.19638Z" fill="#FBFBF6"/>
</svg>

</i>
            </a>
            <a class="rounded-full border border-white/20 w-10 h-10 flex items-center justify-center" target="_blank" href="https://www.linkedin.com/company/first-round-capital/">
                <i class="icon icon-linkedin fill-current visible w-4 h-4 text-body-bg">
  <svg width="14" height="13" viewBox="0 0 14 13" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M9.59995 4.28955C10.6343 4.28955 11.6263 4.6888 12.3577 5.39946C13.0891 6.11013 13.5 7.07399 13.5 8.07902V12.5001H10.9V8.07902C10.9 7.74401 10.763 7.42273 10.5192 7.18584C10.2754 6.94895 9.94473 6.81587 9.59995 6.81587C9.25517 6.81587 8.92451 6.94895 8.68071 7.18584C8.43692 7.42273 8.29995 7.74401 8.29995 8.07902V12.5001H5.69995V8.07902C5.69995 7.07399 6.11084 6.11013 6.84223 5.39946C7.57363 4.6888 8.56561 4.28955 9.59995 4.28955Z" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M3.1 4.9209H0.5V12.4998H3.1V4.9209Z" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M1.8 3.02632C2.51797 3.02632 3.1 2.46078 3.1 1.76316C3.1 1.06554 2.51797 0.5 1.8 0.5C1.08203 0.5 0.5 1.06554 0.5 1.76316C0.5 2.46078 1.08203 3.02632 1.8 3.02632Z" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/>
</svg>

</i>
            </a>
            <a class="rounded-full border border-white/20 w-10 h-10 flex items-center justify-center" target="_blank" href="https://www.youtube.com/user/FirstRoundCapital">
                <i class="icon icon-youtube fill-current visible w-4 h-4 text-body-bg">
  <svg width="14" height="11" viewBox="0 0 14 11" fill="none" xmlns="http://www.w3.org/2000/svg">
<path fill-rule="evenodd" clip-rule="evenodd" d="M5.62262 7.54814L5.6222 3.55339L9.2247 5.55765L5.62262 7.54814ZM13.533 2.86121C13.533 2.86121 13.4026 1.88151 13.003 1.45009C12.4959 0.883601 11.9276 0.880935 11.6672 0.848057C9.80134 0.704102 7.00259 0.704102 7.00259 0.704102H6.99676C6.99676 0.704102 4.19801 0.704102 2.33217 0.848057C2.07134 0.880935 1.50342 0.883601 0.995924 1.45009C0.596341 1.88151 0.466341 2.86121 0.466341 2.86121C0.466341 2.86121 0.333008 4.01196 0.333008 5.16227V6.24105C0.333008 7.3918 0.466341 8.54211 0.466341 8.54211C0.466341 8.54211 0.596341 9.5218 0.995924 9.95322C1.50342 10.5197 2.16967 10.5019 2.46634 10.561C3.53301 10.6703 6.99967 10.7041 6.99967 10.7041C6.99967 10.7041 9.80134 10.6997 11.6672 10.5557C11.9276 10.5224 12.4959 10.5197 13.003 9.95322C13.4026 9.5218 13.533 8.54211 13.533 8.54211C13.533 8.54211 13.6663 7.3918 13.6663 6.24105V5.16227C13.6663 4.01196 13.533 2.86121 13.533 2.86121Z" fill="#FBFBF6"/>
</svg>

</i>            </a>
            </div>
          </nav>
      </div>

   </div>

   </div>
</footer>

      <script>
 function addTranscriptButton() {
  const transcriptWrapper = document.querySelector('.transcript');

  if (transcriptWrapper && !transcriptWrapper.querySelector('.transcript-button')) {
    const transcriptButton = document.createElement('button');
    transcriptButton.classList.add('transcript-button');
    transcriptButton.innerText = 'Continue reading';
    transcriptButton.addEventListener('click', function () {
      transcriptWrapper.classList.add('transcript-visible');
    });
    transcriptWrapper.appendChild(transcriptButton);
  }
}

 document.addEventListener('DOMContentLoaded', function () {
  // Try to add button immediately
  addTranscriptButton();

  // If not found immediately, use a single retry with a reasonable delay
  if (!document.querySelector('.transcript .transcript-button')) {
    setTimeout(addTranscriptButton, 300);
  }

  document.querySelectorAll('p').forEach(p => {
    if (p.textContent.trim() === '---') {
      const hr = document.createElement('hr');
      hr.className = 'podcast-divider';
      p.replaceWith(hr);
    }
  });
});

  </script>

     

    

    <script>
  const OutpostPub = {
    apiDomain: "https://reviewfirstroundcom.outpost.pub",
    apiKey: "47028acb-5668-4e2c-af15-5c902acfe90d",
    siteName: "First Round",
    memberId: "",
    memberEmail: "",
        post: {
          id: "6585b26b580f2439a1e0baea",
          tags: ["PR &amp; Marketing", "Women", "Product marketing", "Storytelling", "Brand", "PR", "Brooke Hammerling", "Brew Media Relations", "#wp", "#wp-post", "#Import 2023-12-22 15:59"],
        }
  }
</script>
<script defer src="https://assets.outpostpublishingcoop.com/assets/v2/js/outpost-pub.js"></script>
    <script>
!function(w,d,id,ns,s){var c=w[ns]=w[ns]||{};if(c.ready||(c.q=[],c.ready=function(){c.q.push(arguments)}),!d.getElementById(id)){var e=d.createElement(s);e.id=id,e.defer=true,e.src="https://assets.context.ly/kit/6.latest/loader.js";var h=d.getElementsByTagName(s)[0];h.parentNode.insertBefore(e,h)}}(window,document,"ctx-loader","Contextly","script");
</script>
<script>
Contextly.ready('widgets');
</script>
  </body>
</html>
