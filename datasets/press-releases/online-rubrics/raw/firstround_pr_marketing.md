<!-- URL: https://review.firstround.com/articles/pr-and-marketing/ -->
<!-- HTTP: 200 SIZE:   130731 -->
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">

    <title>PR &amp; Marketing - First Round</title>

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
    currentPage: parseInt('1'),
    nextPage: parseInt('2'),
    nextPageLink: '',
    maxPages: parseInt('6'), 
    lastPage: `1` === `6` ? true : false,
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
    <link rel="icon" href="https://review.firstround.com/content/images/size/w256h256/format/png/2024/03/First-Round-icon-blk.svg" type="image/png">
    <link rel="canonical" href="https://review.firstround.com/articles/pr-and-marketing/">
    <meta name="referrer" content="no-referrer-when-downgrade">
    <link rel="next" href="https://review.firstround.com/articles/pr-and-marketing/page/2/">
    
    <meta property="og:site_name" content="First Round">
    <meta property="og:type" content="website">
    <meta property="og:title" content="PR &amp; Marketing - First Round">
    <meta property="og:description" content="Tactical advice for founders and startup leaders — from interview questions and tips for new managers, to how to find a co-founder and product-market fit.">
    <meta property="og:url" content="https://review.firstround.com/articles/pr-and-marketing/">
    <meta property="og:image" content="https://static.ghost.org/v5.0.0/images/publication-cover.jpg">
    <meta property="article:publisher" content="https://www.facebook.com/firstroundcapital/">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="PR &amp; Marketing - First Round">
    <meta name="twitter:description" content="Tactical advice for founders and startup leaders — from interview questions and tips for new managers, to how to find a co-founder and product-market fit.">
    <meta name="twitter:url" content="https://review.firstround.com/articles/pr-and-marketing/">
    <meta name="twitter:image" content="https://static.ghost.org/v5.0.0/images/publication-cover.jpg">
    <meta name="twitter:site" content="@firstround">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="840">
    
    <script type="application/ld+json">
{
    "@context": "https://schema.org",
    "@type": "Series",
    "publisher": {
        "@type": "Organization",
        "name": "First Round",
        "url": "https://review.firstround.com/",
        "logo": {
            "@type": "ImageObject",
            "url": "https://review.firstround.com/content/images/2024/02/Reviewlogo.png"
        }
    },
    "url": "https://review.firstround.com/articles/pr-and-marketing/",
    "name": "PR & Marketing",
    "mainEntityOfPage": "https://review.firstround.com/articles/pr-and-marketing/"
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
  <body class="tag-template tag-pr-and-marketing relative overflow-x-hidden bg-backgr text-typogr scroll-smooth "
    x-data="{showMenu : false}" :class="{ 'overflow-hidden': showMenu, '': !showMenu}"
    data-user="visitor"
    @keydown.escape="menuOpen = false"
  >

    

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
      <li class="nav-articles nav-current-parent px-4 flex items-center nav-item"
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
      <li class="nav-articles nav-current-parent py-6 px-0 text-left nav-item"
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
      
    <div class="max-w-7xl m-auto mb-8 px-5 lg:px-6 xl:px-5 pt-8">
    <nav class="flex flex-row lg:justify-start space-x-4.5 thin-scroll overflow-y-hidden overflow-x-auto max-lg:pb-6 pb-4">
      <a class="inline-flex items-center border border-black/25 rounded-[60px] text-sm px-6 py-4 min-w-[100px] text-center justify-center h-[48px] basis-auto grow-0 shrink-0 hover:border-black hover:shadow-4xl transition-all duration-300 hover:bg-transparent hover:text-black  " href="/articles/">All</a>
          <a class="inline-flex items-center border border-black/25 rounded-[60px] text-sm px-6 py-4 min-w-[100px] text-center justify-center h-[48px] basis-auto grow-0 shrink-0 hover:border-black hover:shadow-4xl transition-all duration-300 hover:bg-transparent hover:text-black " href="/articles/product/">Product</a>
          <a class="inline-flex items-center border border-black/25 rounded-[60px] text-sm px-6 py-4 min-w-[100px] text-center justify-center h-[48px] basis-auto grow-0 shrink-0 hover:border-black hover:shadow-4xl transition-all duration-300 hover:bg-transparent hover:text-black " href="/articles/fundraising/">Fundraising</a>
          <a class="inline-flex items-center border border-black/25 rounded-[60px] text-sm px-6 py-4 min-w-[100px] text-center justify-center h-[48px] basis-auto grow-0 shrink-0 hover:border-black hover:shadow-4xl transition-all duration-300 hover:bg-transparent hover:text-black " href="/articles/starting-up/">Starting Up</a>
          <a class="inline-flex items-center border border-black/25 rounded-[60px] text-sm px-6 py-4 min-w-[100px] text-center justify-center h-[48px] basis-auto grow-0 shrink-0 hover:border-black hover:shadow-4xl transition-all duration-300 hover:bg-transparent hover:text-black " href="/articles/management/">Management</a>
          <a class="inline-flex items-center border border-black/25 rounded-[60px] text-sm px-6 py-4 min-w-[100px] text-center justify-center h-[48px] basis-auto grow-0 shrink-0 hover:border-black hover:shadow-4xl transition-all duration-300 hover:bg-transparent hover:text-black " href="/articles/engineering/">Engineering</a>
          <a class="inline-flex items-center border border-black/25 rounded-[60px] text-sm px-6 py-4 min-w-[100px] text-center justify-center h-[48px] basis-auto grow-0 shrink-0 hover:border-black hover:shadow-4xl transition-all duration-300 hover:bg-transparent hover:text-black bg-green-200 border-green-200 bg-opacity-75" href="/articles/pr-and-marketing/">PR &amp; Marketing</a>
          <a class="inline-flex items-center border border-black/25 rounded-[60px] text-sm px-6 py-4 min-w-[100px] text-center justify-center h-[48px] basis-auto grow-0 shrink-0 hover:border-black hover:shadow-4xl transition-all duration-300 hover:bg-transparent hover:text-black " href="/articles/must-reads/">Must-reads</a>
          <a class="inline-flex items-center border border-black/25 rounded-[60px] text-sm px-6 py-4 min-w-[100px] text-center justify-center h-[48px] basis-auto grow-0 shrink-0 hover:border-black hover:shadow-4xl transition-all duration-300 hover:bg-transparent hover:text-black " href="/articles/people-and-culture/">People &amp; Culture</a>
          <a class="inline-flex items-center border border-black/25 rounded-[60px] text-sm px-6 py-4 min-w-[100px] text-center justify-center h-[48px] basis-auto grow-0 shrink-0 hover:border-black hover:shadow-4xl transition-all duration-300 hover:bg-transparent hover:text-black " href="/articles/design/">Design</a>
          <a class="inline-flex items-center border border-black/25 rounded-[60px] text-sm px-6 py-4 min-w-[100px] text-center justify-center h-[48px] basis-auto grow-0 shrink-0 hover:border-black hover:shadow-4xl transition-all duration-300 hover:bg-transparent hover:text-black " href="/articles/sales/">Sales</a>
    </nav>
  </div>
  <div class="pt-4 lg:pt-4 mb-10">
    <h1 class="w-full px-4 text-2xl text-center font-semibold tracking-wide text-black xl:text-3xl ">
      All of our <span ">PR &amp; Marketing</span> articles
    </h1>
  </div>

<div class="m-auto max-w-7xl mb-15 lg:mb-20 px-5 lg:px-6 xl:px-5 relative">

  <div data-post-feed data-label="Articles" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-5 lg:gap-8 firstround-post-items">
        <article data-post-card-wrapper class="w-full firstound-post-item relative max-md:border-b max-md:border-gray-100 max-md:pb-5 max-md:last:border-none">
          <div class="flex flex-col w-full cursor-pointer card-to-responsive">
    <a href="/how-to-pick-company-name/" class="max-md:order-2">
    <figure class="overflow-hidden mb-5 max-md:order-2 post-f-img">
      <picture  >
  <source 
    srcset="/content/images/size/w320/format/webp/2025/11/NamingHeader.png 320w, /content/images/size/w640/format/webp/2025/11/NamingHeader.png 600w, /content/images/size/w960/format/webp/2025/11/NamingHeader.png 960w, /content/images/size/w1280/format/webp/2025/11/NamingHeader.png 1200w"
    sizes="296px"
    type="image/webp"
  >
  <img class="w-full h-full object-cover aspect-3/2 transition duration-400 blur-sm"
    onload="this.classList.remove('blur-sm')"
    loading="lazy"
    srcset="/content/images/size/w320/format/webp/2025/11/NamingHeader.png 320w, /content/images/size/w640/format/webp/2025/11/NamingHeader.png 600w, /content/images/size/w960/format/webp/2025/11/NamingHeader.png 960w, /content/images/size/w1280/format/webp/2025/11/NamingHeader.png 1200w"
    sizes="296px"
    src="/content/images/size/w30/2025/11/NamingHeader.png"
    alt="What’s in a Name? A Lot, Actually. Here’s How to Pick the Right One For Your Company"
  />
</picture>    </figure>
    </a>

  <div data-label="Info" class="flex flex-col divider-y items-start">


        <a href="/articles/starting-up/" data-label="PrimaryCategory" class="firstround-tag max-md:!hidden">Starting Up</a>


    <a href="/how-to-pick-company-name/" title="What’s in a Name? A Lot, Actually. Here’s How to Pick the Right One For Your Company">
      <h3 data-label="Title" title="What’s in a Name? A Lot, Actually. Here’s How to Pick the Right One For Your Company" class="!font-skandia-medium text-xl tracking-tight line-clamp-2 mt-2.5 mb-2.5">What’s in a Name? A Lot, Actually. Here’s How to Pick the Right One For Your Company</h3>
        <p data-label="Description" class="text-base line-clamp-4 max-md:!hidden  leading-[1.4]">A playbook for picking a name that actually fits your company (and all the logistics that come with naming you may not be thinking about).</p>
     </a>
  </div>
</div>        </article>
        <article data-post-card-wrapper class="w-full firstound-post-item relative max-md:border-b max-md:border-gray-100 max-md:pb-5 max-md:last:border-none">
          <div class="flex flex-col w-full cursor-pointer card-to-responsive">
    <a href="/how-to-launch-your-startup-out-of-stealth/" class="max-md:order-2">
    <figure class="overflow-hidden mb-5 max-md:order-2 post-f-img">
      <picture  >
  <source 
    srcset="/content/images/size/w320/format/webp/2025/08/FirstRound_Claire-Butler_Focus_LI.png 320w, /content/images/size/w640/format/webp/2025/08/FirstRound_Claire-Butler_Focus_LI.png 600w, /content/images/size/w960/format/webp/2025/08/FirstRound_Claire-Butler_Focus_LI.png 960w, /content/images/size/w1280/format/webp/2025/08/FirstRound_Claire-Butler_Focus_LI.png 1200w"
    sizes="296px"
    type="image/webp"
  >
  <img class="w-full h-full object-cover aspect-3/2 transition duration-400 blur-sm"
    onload="this.classList.remove('blur-sm')"
    loading="lazy"
    srcset="/content/images/size/w320/format/webp/2025/08/FirstRound_Claire-Butler_Focus_LI.png 320w, /content/images/size/w640/format/webp/2025/08/FirstRound_Claire-Butler_Focus_LI.png 600w, /content/images/size/w960/format/webp/2025/08/FirstRound_Claire-Butler_Focus_LI.png 960w, /content/images/size/w1280/format/webp/2025/08/FirstRound_Claire-Butler_Focus_LI.png 1200w"
    sizes="296px"
    src="/content/images/size/w30/2025/08/FirstRound_Claire-Butler_Focus_LI.png"
    alt="How to Launch Your Startup Out of Stealth, from Figma&#x27;s First Marketer"
  />
</picture>    </figure>
    </a>

  <div data-label="Info" class="flex flex-col divider-y items-start">


        <a href="/articles/pr-and-marketing/" data-label="PrimaryCategory" class="firstround-tag max-md:!hidden">PR &amp; Marketing</a>


    <a href="/how-to-launch-your-startup-out-of-stealth/" title="How to Launch Your Startup Out of Stealth, from Figma&#x27;s First Marketer">
      <h3 data-label="Title" title="How to Launch Your Startup Out of Stealth, from Figma&#x27;s First Marketer" class="!font-skandia-medium text-xl tracking-tight line-clamp-2 mt-2.5 mb-2.5">How to Launch Your Startup Out of Stealth, from Figma&#x27;s First Marketer</h3>
        <p data-label="Description" class="text-base line-clamp-4 max-md:!hidden  leading-[1.4]">Claire Butler joined Figma six months before its public launch and helped bring the design product out into the world. Here, she defangs the launch process, outlining all the steps you need to take to get ready now.</p>
     </a>
  </div>
</div>        </article>
        <article data-post-card-wrapper class="w-full firstound-post-item relative max-md:border-b max-md:border-gray-100 max-md:pb-5 max-md:last:border-none">
          <div class="flex flex-col w-full cursor-pointer card-to-responsive">
    <a href="/how-to-adapt-your-pitch-deck-into-your-website/" class="max-md:order-2">
    <figure class="overflow-hidden mb-5 max-md:order-2 post-f-img">
      <picture  >
  <source 
    srcset="/content/images/size/w320/format/webp/2025/08/rocks-1.png 320w, /content/images/size/w640/format/webp/2025/08/rocks-1.png 600w, /content/images/size/w960/format/webp/2025/08/rocks-1.png 960w, /content/images/size/w1280/format/webp/2025/08/rocks-1.png 1200w"
    sizes="296px"
    type="image/webp"
  >
  <img class="w-full h-full object-cover aspect-3/2 transition duration-400 blur-sm"
    onload="this.classList.remove('blur-sm')"
    loading="lazy"
    srcset="/content/images/size/w320/format/webp/2025/08/rocks-1.png 320w, /content/images/size/w640/format/webp/2025/08/rocks-1.png 600w, /content/images/size/w960/format/webp/2025/08/rocks-1.png 960w, /content/images/size/w1280/format/webp/2025/08/rocks-1.png 1200w"
    sizes="296px"
    src="/content/images/size/w30/2025/08/rocks-1.png"
    alt="Don’t Ship Your Pitch Deck to Your Website. Here’s How to Translate Your Story for Customers"
  />
</picture>    </figure>
    </a>

  <div data-label="Info" class="flex flex-col divider-y items-start">


        <a href="/articles/pr-and-marketing/" data-label="PrimaryCategory" class="firstround-tag max-md:!hidden">PR &amp; Marketing</a>


    <a href="/how-to-adapt-your-pitch-deck-into-your-website/" title="Don’t Ship Your Pitch Deck to Your Website. Here’s How to Translate Your Story for Customers">
      <h3 data-label="Title" title="Don’t Ship Your Pitch Deck to Your Website. Here’s How to Translate Your Story for Customers" class="!font-skandia-medium text-xl tracking-tight line-clamp-2 mt-2.5 mb-2.5">Don’t Ship Your Pitch Deck to Your Website. Here’s How to Translate Your Story for Customers</h3>
        <p data-label="Description" class="text-base line-clamp-4 max-md:!hidden  leading-[1.4]">Your customers don’t want to read your pitch deck. Marketing expert Emily Kramer explains how to turn your investor talking points into your website messaging.</p>
     </a>
  </div>
</div>        </article>
        <article data-post-card-wrapper class="w-full firstound-post-item relative max-md:border-b max-md:border-gray-100 max-md:pb-5 max-md:last:border-none">
          <div class="flex flex-col w-full cursor-pointer card-to-responsive">
    <a href="/9-marketing-skills-startups-want-on-their-teams/" class="max-md:order-2">
    <figure class="overflow-hidden mb-5 max-md:order-2 post-f-img">
      <picture  >
  <source 
    srcset="/content/images/size/w320/format/webp/2024/10/GettyImages-1785783984.jpg 320w, /content/images/size/w640/format/webp/2024/10/GettyImages-1785783984.jpg 600w, /content/images/size/w960/format/webp/2024/10/GettyImages-1785783984.jpg 960w, /content/images/size/w1280/format/webp/2024/10/GettyImages-1785783984.jpg 1200w"
    sizes="296px"
    type="image/webp"
  >
  <img class="w-full h-full object-cover aspect-3/2 transition duration-400 blur-sm"
    onload="this.classList.remove('blur-sm')"
    loading="lazy"
    srcset="/content/images/size/w320/format/webp/2024/10/GettyImages-1785783984.jpg 320w, /content/images/size/w640/format/webp/2024/10/GettyImages-1785783984.jpg 600w, /content/images/size/w960/format/webp/2024/10/GettyImages-1785783984.jpg 960w, /content/images/size/w1280/format/webp/2024/10/GettyImages-1785783984.jpg 1200w"
    sizes="296px"
    src="/content/images/size/w30/2024/10/GettyImages-1785783984.jpg"
    alt="9 Marketing Skills Startups Want on Their Teams"
  />
</picture>    </figure>
    </a>

  <div data-label="Info" class="flex flex-col divider-y items-start">


        <a href="/articles/pr-and-marketing/" data-label="PrimaryCategory" class="firstround-tag max-md:!hidden">PR &amp; Marketing</a>


    <a href="/9-marketing-skills-startups-want-on-their-teams/" title="9 Marketing Skills Startups Want on Their Teams">
      <h3 data-label="Title" title="9 Marketing Skills Startups Want on Their Teams" class="!font-skandia-medium text-xl tracking-tight line-clamp-2 mt-2.5 mb-2.5">9 Marketing Skills Startups Want on Their Teams</h3>
        <p data-label="Description" class="text-base line-clamp-4 max-md:!hidden  leading-[1.4]">Find out the skills needed for a marketer to succeed and help companies reach their full potential with this complete list.</p>
     </a>
  </div>
</div>        </article>
        <article data-post-card-wrapper class="w-full firstound-post-item relative max-md:border-b max-md:border-gray-100 max-md:pb-5 max-md:last:border-none">
          <div class="flex flex-col w-full cursor-pointer card-to-responsive">
    <a href="/passing-the-button-clicker-test-and-other-0-to-1-lessons-from-lattices-first-marketing-hire/" class="max-md:order-2">
    <figure class="overflow-hidden mb-5 max-md:order-2 post-f-img">
      <picture  >
  <source 
    srcset="/content/images/size/w320/format/webp/2024/09/Megaphone-cursor-final-3.jpg 320w, /content/images/size/w640/format/webp/2024/09/Megaphone-cursor-final-3.jpg 600w, /content/images/size/w960/format/webp/2024/09/Megaphone-cursor-final-3.jpg 960w, /content/images/size/w1280/format/webp/2024/09/Megaphone-cursor-final-3.jpg 1200w"
    sizes="296px"
    type="image/webp"
  >
  <img class="w-full h-full object-cover aspect-3/2 transition duration-400 blur-sm"
    onload="this.classList.remove('blur-sm')"
    loading="lazy"
    srcset="/content/images/size/w320/format/webp/2024/09/Megaphone-cursor-final-3.jpg 320w, /content/images/size/w640/format/webp/2024/09/Megaphone-cursor-final-3.jpg 600w, /content/images/size/w960/format/webp/2024/09/Megaphone-cursor-final-3.jpg 960w, /content/images/size/w1280/format/webp/2024/09/Megaphone-cursor-final-3.jpg 1200w"
    sizes="296px"
    src="/content/images/size/w30/2024/09/Megaphone-cursor-final-3.jpg"
    alt="Passing the Button-Clicker Test, and Other 0 to 1 Lessons from Lattice’s First Marketing Hire"
  />
</picture>    </figure>
    </a>

  <div data-label="Info" class="flex flex-col divider-y items-start">


        <a href="/articles/pr-and-marketing/" data-label="PrimaryCategory" class="firstround-tag max-md:!hidden">PR &amp; Marketing</a>


    <a href="/passing-the-button-clicker-test-and-other-0-to-1-lessons-from-lattices-first-marketing-hire/" title="Passing the Button-Clicker Test, and Other 0 to 1 Lessons from Lattice’s First Marketing Hire">
      <h3 data-label="Title" title="Passing the Button-Clicker Test, and Other 0 to 1 Lessons from Lattice’s First Marketing Hire" class="!font-skandia-medium text-xl tracking-tight line-clamp-2 mt-2.5 mb-2.5">Passing the Button-Clicker Test, and Other 0 to 1 Lessons from Lattice’s First Marketing Hire</h3>
        <p data-label="Description" class="text-base line-clamp-4 max-md:!hidden  leading-[1.4]">The behind-the-scenes story of how Lattice built its marketing strategy from the ground up, as told by its third employee and first marketing hire, Alex Kracov.</p>
     </a>
  </div>
</div>        </article>
        <article data-post-card-wrapper class="w-full firstound-post-item relative max-md:border-b max-md:border-gray-100 max-md:pb-5 max-md:last:border-none">
          <div class="flex flex-col w-full cursor-pointer card-to-responsive">
    <a href="/why-startup-marketers-should-be-diagnosticians-advice-from-stripe-and-openais-first-marketing-hire/" class="max-md:order-2">
    <figure class="overflow-hidden mb-5 max-md:order-2 post-f-img">
      <picture  >
  <source 
    srcset="/content/images/size/w320/format/webp/2024/08/GettyImages-88160315-2.jpg 320w, /content/images/size/w640/format/webp/2024/08/GettyImages-88160315-2.jpg 600w, /content/images/size/w960/format/webp/2024/08/GettyImages-88160315-2.jpg 960w, /content/images/size/w1280/format/webp/2024/08/GettyImages-88160315-2.jpg 1200w"
    sizes="296px"
    type="image/webp"
  >
  <img class="w-full h-full object-cover aspect-3/2 transition duration-400 blur-sm"
    onload="this.classList.remove('blur-sm')"
    loading="lazy"
    srcset="/content/images/size/w320/format/webp/2024/08/GettyImages-88160315-2.jpg 320w, /content/images/size/w640/format/webp/2024/08/GettyImages-88160315-2.jpg 600w, /content/images/size/w960/format/webp/2024/08/GettyImages-88160315-2.jpg 960w, /content/images/size/w1280/format/webp/2024/08/GettyImages-88160315-2.jpg 1200w"
    sizes="296px"
    src="/content/images/size/w30/2024/08/GettyImages-88160315-2.jpg"
    alt="Why Startup Marketers Should Be Diagnosticians — Advice from OpenAI&#x27;s and Stripe&#x27;s First Marketing Hire"
  />
</picture>    </figure>
    </a>

  <div data-label="Info" class="flex flex-col divider-y items-start">


        <a href="/articles/pr-and-marketing/" data-label="PrimaryCategory" class="firstround-tag max-md:!hidden">PR &amp; Marketing</a>


    <a href="/why-startup-marketers-should-be-diagnosticians-advice-from-stripe-and-openais-first-marketing-hire/" title="Why Startup Marketers Should Be Diagnosticians — Advice from OpenAI&#x27;s and Stripe&#x27;s First Marketing Hire">
      <h3 data-label="Title" title="Why Startup Marketers Should Be Diagnosticians — Advice from OpenAI&#x27;s and Stripe&#x27;s First Marketing Hire" class="!font-skandia-medium text-xl tracking-tight line-clamp-2 mt-2.5 mb-2.5">Why Startup Marketers Should Be Diagnosticians — Advice from OpenAI&#x27;s and Stripe&#x27;s First Marketing Hire</h3>
        <p data-label="Description" class="text-base line-clamp-4 max-md:!hidden  leading-[1.4]">Krithika Muthukumar was the very first marketer at OpenAI and Stripe. Here, she unpacks what it takes to found and scale a marketing operation to build an enduring brand.</p>
     </a>
  </div>
</div>        </article>
        <article data-post-card-wrapper class="w-full firstound-post-item relative max-md:border-b max-md:border-gray-100 max-md:pb-5 max-md:last:border-none">
          <div class="flex flex-col w-full cursor-pointer card-to-responsive">
    <a href="/how-notion-does-marketing-a-deep-dive-into-its-community-influencers-growth-playbooks/" class="max-md:order-2">
    <figure class="overflow-hidden mb-5 max-md:order-2 post-f-img">
      <picture  >
  <source 
    srcset="/content/images/size/w320/format/webp/2024/04/Review-FRCNotionhero.png 320w, /content/images/size/w640/format/webp/2024/04/Review-FRCNotionhero.png 600w, /content/images/size/w960/format/webp/2024/04/Review-FRCNotionhero.png 960w, /content/images/size/w1280/format/webp/2024/04/Review-FRCNotionhero.png 1200w"
    sizes="296px"
    type="image/webp"
  >
  <img class="w-full h-full object-cover aspect-3/2 transition duration-400 blur-sm"
    onload="this.classList.remove('blur-sm')"
    loading="lazy"
    srcset="/content/images/size/w320/format/webp/2024/04/Review-FRCNotionhero.png 320w, /content/images/size/w640/format/webp/2024/04/Review-FRCNotionhero.png 600w, /content/images/size/w960/format/webp/2024/04/Review-FRCNotionhero.png 960w, /content/images/size/w1280/format/webp/2024/04/Review-FRCNotionhero.png 1200w"
    sizes="296px"
    src="/content/images/size/w30/2024/04/Review-FRCNotionhero.png"
    alt="How Notion Does Marketing: A Deep-Dive Into its Community, Influencers &amp; Growth Playbooks"
  />
</picture>    </figure>
    </a>

  <div data-label="Info" class="flex flex-col divider-y items-start">


        <a href="/articles/pr-and-marketing/" data-label="PrimaryCategory" class="firstround-tag max-md:!hidden">PR &amp; Marketing</a>


    <a href="/how-notion-does-marketing-a-deep-dive-into-its-community-influencers-growth-playbooks/" title="How Notion Does Marketing: A Deep-Dive Into its Community, Influencers &amp; Growth Playbooks">
      <h3 data-label="Title" title="How Notion Does Marketing: A Deep-Dive Into its Community, Influencers &amp; Growth Playbooks" class="!font-skandia-medium text-xl tracking-tight line-clamp-2 mt-2.5 mb-2.5">How Notion Does Marketing: A Deep-Dive Into its Community, Influencers &amp; Growth Playbooks</h3>
        <p data-label="Description" class="text-base line-clamp-4 max-md:!hidden  leading-[1.4]">We peel back the layers of Notion’s multifaceted marketing strategy, diving deep into three distinct channels that have helped power its meteoric rise: community, influencer marketing, and growth and analytics.</p>
     </a>
  </div>
</div>        </article>
        <article data-post-card-wrapper class="w-full firstound-post-item relative max-md:border-b max-md:border-gray-100 max-md:pb-5 max-md:last:border-none">
          <div class="flex flex-col w-full cursor-pointer card-to-responsive">
    <a href="/how-to-build-an-iconic-brand-without-breaking-the-bank/" class="max-md:order-2">
    <figure class="overflow-hidden mb-5 max-md:order-2 post-f-img">
      <picture  >
  <source 
    srcset="/content/images/size/w320/format/webp/2024/02/GettyImages-1436245700.jpg 320w, /content/images/size/w640/format/webp/2024/02/GettyImages-1436245700.jpg 600w, /content/images/size/w960/format/webp/2024/02/GettyImages-1436245700.jpg 960w, /content/images/size/w1280/format/webp/2024/02/GettyImages-1436245700.jpg 1200w"
    sizes="296px"
    type="image/webp"
  >
  <img class="w-full h-full object-cover aspect-3/2 transition duration-400 blur-sm"
    onload="this.classList.remove('blur-sm')"
    loading="lazy"
    srcset="/content/images/size/w320/format/webp/2024/02/GettyImages-1436245700.jpg 320w, /content/images/size/w640/format/webp/2024/02/GettyImages-1436245700.jpg 600w, /content/images/size/w960/format/webp/2024/02/GettyImages-1436245700.jpg 960w, /content/images/size/w1280/format/webp/2024/02/GettyImages-1436245700.jpg 1200w"
    sizes="296px"
    src="/content/images/size/w30/2024/02/GettyImages-1436245700.jpg"
    alt="How to Build an Iconic Brand Without Breaking the Bank — This Founder&#x27;s Formula for Early Marketing"
  />
</picture>    </figure>
    </a>

  <div data-label="Info" class="flex flex-col divider-y items-start">


        <a href="/articles/pr-and-marketing/" data-label="PrimaryCategory" class="firstround-tag max-md:!hidden">PR &amp; Marketing</a>


    <a href="/how-to-build-an-iconic-brand-without-breaking-the-bank/" title="How to Build an Iconic Brand Without Breaking the Bank — This Founder&#x27;s Formula for Early Marketing">
      <h3 data-label="Title" title="How to Build an Iconic Brand Without Breaking the Bank — This Founder&#x27;s Formula for Early Marketing" class="!font-skandia-medium text-xl tracking-tight line-clamp-2 mt-2.5 mb-2.5">How to Build an Iconic Brand Without Breaking the Bank — This Founder&#x27;s Formula for Early Marketing</h3>
        <p data-label="Description" class="text-base line-clamp-4 max-md:!hidden  leading-[1.4]">Studs co-founder Lisa Bubbers shares how they melded a strong brand with operational rigor to build a thriving business with minimal marketing expenses.</p>
     </a>
  </div>
</div>        </article>
        <article data-post-card-wrapper class="w-full firstound-post-item relative max-md:border-b max-md:border-gray-100 max-md:pb-5 max-md:last:border-none">
          <div class="flex flex-col w-full cursor-pointer card-to-responsive">
    <a href="/storytelling-as-a-craft-advice-from-5-experts-on-how-to-tell-a-compelling-business-story/" class="max-md:order-2">
    <figure class="overflow-hidden mb-5 max-md:order-2 post-f-img">
      <picture  >
  <source 
    srcset="/content/images/size/w320/format/webp/firstround-2fgettyimages-1158477464.jpg 320w, /content/images/size/w640/format/webp/firstround-2fgettyimages-1158477464.jpg 600w, /content/images/size/w960/format/webp/firstround-2fgettyimages-1158477464.jpg 960w, /content/images/size/w1280/format/webp/firstround-2fgettyimages-1158477464.jpg 1200w"
    sizes="296px"
    type="image/webp"
  >
  <img class="w-full h-full object-cover aspect-3/2 transition duration-400 blur-sm"
    onload="this.classList.remove('blur-sm')"
    loading="lazy"
    srcset="/content/images/size/w320/format/webp/firstround-2fgettyimages-1158477464.jpg 320w, /content/images/size/w640/format/webp/firstround-2fgettyimages-1158477464.jpg 600w, /content/images/size/w960/format/webp/firstround-2fgettyimages-1158477464.jpg 960w, /content/images/size/w1280/format/webp/firstround-2fgettyimages-1158477464.jpg 1200w"
    sizes="296px"
    src="/content/images/size/w30/firstround-2fgettyimages-1158477464.jpg"
    alt="Storytelling as a Craft: Advice from 5 Experts on How to Tell a Compelling Business Story"
  />
</picture>    </figure>
    </a>

  <div data-label="Info" class="flex flex-col divider-y items-start">


        <a href="/articles/pr-and-marketing/" data-label="PrimaryCategory" class="firstround-tag max-md:!hidden">PR &amp; Marketing</a>


    <a href="/storytelling-as-a-craft-advice-from-5-experts-on-how-to-tell-a-compelling-business-story/" title="Storytelling as a Craft: Advice from 5 Experts on How to Tell a Compelling Business Story">
      <h3 data-label="Title" title="Storytelling as a Craft: Advice from 5 Experts on How to Tell a Compelling Business Story" class="!font-skandia-medium text-xl tracking-tight line-clamp-2 mt-2.5 mb-2.5">Storytelling as a Craft: Advice from 5 Experts on How to Tell a Compelling Business Story</h3>
        <p data-label="Description" class="text-base line-clamp-4 max-md:!hidden  leading-[1.4]">Every startup leader should practice flexing their storytelling muscles. To help warm these muscles up, we’ve rounded up some of our best advice about storytelling in business that’s been featured in the Review.</p>
     </a>
  </div>
</div>        </article>
        <article data-post-card-wrapper class="w-full firstound-post-item relative max-md:border-b max-md:border-gray-100 max-md:pb-5 max-md:last:border-none">
          <div class="flex flex-col w-full cursor-pointer card-to-responsive">
    <a href="/a-founders-framework-for-understanding-performance-vs-brand-marketing/" class="max-md:order-2">
    <figure class="overflow-hidden mb-5 max-md:order-2 post-f-img">
      <picture  >
  <source 
    srcset="/content/images/size/w320/format/webp/firstround-2fgettyimages-1158571584.jpg 320w, /content/images/size/w640/format/webp/firstround-2fgettyimages-1158571584.jpg 600w, /content/images/size/w960/format/webp/firstround-2fgettyimages-1158571584.jpg 960w, /content/images/size/w1280/format/webp/firstround-2fgettyimages-1158571584.jpg 1200w"
    sizes="296px"
    type="image/webp"
  >
  <img class="w-full h-full object-cover aspect-3/2 transition duration-400 blur-sm"
    onload="this.classList.remove('blur-sm')"
    loading="lazy"
    srcset="/content/images/size/w320/format/webp/firstround-2fgettyimages-1158571584.jpg 320w, /content/images/size/w640/format/webp/firstround-2fgettyimages-1158571584.jpg 600w, /content/images/size/w960/format/webp/firstround-2fgettyimages-1158571584.jpg 960w, /content/images/size/w1280/format/webp/firstround-2fgettyimages-1158571584.jpg 1200w"
    sizes="296px"
    src="/content/images/size/w30/firstround-2fgettyimages-1158571584.jpg"
    alt="A Founder’s Framework for Understanding Performance vs. Brand Marketing for Your Startup"
  />
</picture>    </figure>
    </a>

  <div data-label="Info" class="flex flex-col divider-y items-start">


        <a href="/articles/pr-and-marketing/" data-label="PrimaryCategory" class="firstround-tag max-md:!hidden">PR &amp; Marketing</a>


    <a href="/a-founders-framework-for-understanding-performance-vs-brand-marketing/" title="A Founder’s Framework for Understanding Performance vs. Brand Marketing for Your Startup">
      <h3 data-label="Title" title="A Founder’s Framework for Understanding Performance vs. Brand Marketing for Your Startup" class="!font-skandia-medium text-xl tracking-tight line-clamp-2 mt-2.5 mb-2.5">A Founder’s Framework for Understanding Performance vs. Brand Marketing for Your Startup</h3>
        <p data-label="Description" class="text-base line-clamp-4 max-md:!hidden  leading-[1.4]">Startup marketing experts Sarah Emmott and Holly Chen share a multi-step process for how to craft a brand and performance marketing strategy, including how to set goals, allocate the right budget, and measure success.</p>
     </a>
  </div>
</div>        </article>
        <article data-post-card-wrapper class="w-full firstound-post-item relative max-md:border-b max-md:border-gray-100 max-md:pb-5 max-md:last:border-none">
          <div class="flex flex-col w-full cursor-pointer card-to-responsive">
    <a href="/unlock-the-power-of-money-words-how-founders-can-write-copy-that-converts/" class="max-md:order-2">
    <figure class="overflow-hidden mb-5 max-md:order-2 post-f-img">
      <picture  >
  <source 
    srcset="/content/images/size/w320/format/webp/firstround-2fgettyimages-120228953.jpg 320w, /content/images/size/w640/format/webp/firstround-2fgettyimages-120228953.jpg 600w, /content/images/size/w960/format/webp/firstround-2fgettyimages-120228953.jpg 960w, /content/images/size/w1280/format/webp/firstround-2fgettyimages-120228953.jpg 1200w"
    sizes="296px"
    type="image/webp"
  >
  <img class="w-full h-full object-cover aspect-3/2 transition duration-400 blur-sm"
    onload="this.classList.remove('blur-sm')"
    loading="lazy"
    srcset="/content/images/size/w320/format/webp/firstround-2fgettyimages-120228953.jpg 320w, /content/images/size/w640/format/webp/firstround-2fgettyimages-120228953.jpg 600w, /content/images/size/w960/format/webp/firstround-2fgettyimages-120228953.jpg 960w, /content/images/size/w1280/format/webp/firstround-2fgettyimages-120228953.jpg 1200w"
    sizes="296px"
    src="/content/images/size/w30/firstround-2fgettyimages-120228953.jpg"
    alt="Unlock the Power of Money Words: How Founders Can Write Copy That Converts"
  />
</picture>    </figure>
    </a>

  <div data-label="Info" class="flex flex-col divider-y items-start">


        <a href="/articles/pr-and-marketing/" data-label="PrimaryCategory" class="firstround-tag max-md:!hidden">PR &amp; Marketing</a>


    <a href="/unlock-the-power-of-money-words-how-founders-can-write-copy-that-converts/" title="Unlock the Power of Money Words: How Founders Can Write Copy That Converts">
      <h3 data-label="Title" title="Unlock the Power of Money Words: How Founders Can Write Copy That Converts" class="!font-skandia-medium text-xl tracking-tight line-clamp-2 mt-2.5 mb-2.5">Unlock the Power of Money Words: How Founders Can Write Copy That Converts</h3>
        <p data-label="Description" class="text-base line-clamp-4 max-md:!hidden  leading-[1.4]">Copyhackers’ Joanna Wiebe shares the science behind word choice to help take your copy from guesswork to goldmine.</p>
     </a>
  </div>
</div>        </article>
        <article data-post-card-wrapper class="w-full firstound-post-item relative max-md:border-b max-md:border-gray-100 max-md:pb-5 max-md:last:border-none">
          <div class="flex flex-col w-full cursor-pointer card-to-responsive">
    <a href="/the-5-phases-of-figmas-community-led-growth-from-stealth-to-enterprise/" class="max-md:order-2">
    <figure class="overflow-hidden mb-5 max-md:order-2 post-f-img">
      <picture  >
  <source 
    srcset="/content/images/size/w320/format/webp/firstround-2fgettyimages-1352105257.jpg 320w, /content/images/size/w640/format/webp/firstround-2fgettyimages-1352105257.jpg 600w, /content/images/size/w960/format/webp/firstround-2fgettyimages-1352105257.jpg 960w, /content/images/size/w1280/format/webp/firstround-2fgettyimages-1352105257.jpg 1200w"
    sizes="296px"
    type="image/webp"
  >
  <img class="w-full h-full object-cover aspect-3/2 transition duration-400 blur-sm"
    onload="this.classList.remove('blur-sm')"
    loading="lazy"
    srcset="/content/images/size/w320/format/webp/firstround-2fgettyimages-1352105257.jpg 320w, /content/images/size/w640/format/webp/firstround-2fgettyimages-1352105257.jpg 600w, /content/images/size/w960/format/webp/firstround-2fgettyimages-1352105257.jpg 960w, /content/images/size/w1280/format/webp/firstround-2fgettyimages-1352105257.jpg 1200w"
    sizes="296px"
    src="/content/images/size/w30/firstround-2fgettyimages-1352105257.jpg"
    alt="The 5 Phases of Figma’s Community-Led Growth: From Stealth to Enterprise"
  />
</picture>    </figure>
    </a>

  <div data-label="Info" class="flex flex-col divider-y items-start">


        <a href="/articles/pr-and-marketing/" data-label="PrimaryCategory" class="firstround-tag max-md:!hidden">PR &amp; Marketing</a>


    <a href="/the-5-phases-of-figmas-community-led-growth-from-stealth-to-enterprise/" title="The 5 Phases of Figma’s Community-Led Growth: From Stealth to Enterprise">
      <h3 data-label="Title" title="The 5 Phases of Figma’s Community-Led Growth: From Stealth to Enterprise" class="!font-skandia-medium text-xl tracking-tight line-clamp-2 mt-2.5 mb-2.5">The 5 Phases of Figma’s Community-Led Growth: From Stealth to Enterprise</h3>
        <p data-label="Description" class="text-base line-clamp-4 max-md:!hidden  leading-[1.4]">Claire Butler, Figma’s first business hire, walks us through the different phases of the company’s community-led growth model — from planting the seeds with power users in stealth mode all the way to layering a sales motion and moving upmarket.</p>
     </a>
  </div>
</div>        </article>
  </div>

</div>

<script>
  window.addEventListener('scroll', (event) => {
    if (isInViewport('.footer')) loadMorePosts();
  });
</script>

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
      <li class="nav-articles nav-current-parent  nav-item"
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
  const OutpostPub = {
    apiDomain: "https://reviewfirstroundcom.outpost.pub",
    apiKey: "47028acb-5668-4e2c-af15-5c902acfe90d",
    siteName: "First Round",
    memberId: "",
    memberEmail: "",
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
