<!-- URL: https://review.firstround.com/the-top-comms-mistakes-startups-make-and-how-to-avoid-them/ -->
<!-- HTTP: 200 SIZE:   140652 -->
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">

    <title>The Top Comms Mistakes Startups Make — And How To Avoid Them</title>

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
    <meta name="description" content="After two decades of comms experience with companies like Eventbrite, Yahoo, Mattel and Nike, Terra Carmichael shares four common PR mistakes and her techniques for sidestepping them.">
    <link rel="icon" href="https://review.firstround.com/content/images/size/w256h256/format/png/2024/03/First-Round-icon-blk.svg" type="image/png">
    <link rel="canonical" href="https://review.firstround.com/the-top-comms-mistakes-startups-make-and-how-to-avoid-them/">
    <meta name="referrer" content="no-referrer-when-downgrade">
    
    <meta property="og:site_name" content="First Round">
    <meta property="og:type" content="article">
    <meta property="og:title" content="The Top Comms Mistakes Startups Make — And How To Avoid Them">
    <meta property="og:description" content="After two decades of comms experience with companies like Eventbrite, Yahoo, Mattel and Nike, Terra Carmichael shares four common PR mistakes and her techniques for sidestepping them.">
    <meta property="og:url" content="https://review.firstround.com/the-top-comms-mistakes-startups-make-and-how-to-avoid-them/">
    <meta property="og:image" content="https://review.firstround.com/content/images/size/w1200/2056/firstround-2fibmwdxvbqfe9a6toh8we_vntr1912.jpg">
    <meta property="article:published_time" content="2018-07-20T22:54:10.000Z">
    <meta property="article:modified_time" content="2024-03-06T12:42:11.000Z">
    <meta property="article:tag" content="PR &amp;amp; Marketing">
    <meta property="article:tag" content="Sales">
    <meta property="article:tag" content="People &amp;amp; Culture">
    <meta property="article:tag" content="Women">
    <meta property="article:tag" content="Content marketing">
    <meta property="article:tag" content="PR">
    <meta property="article:tag" content="Storytelling">
    <meta property="article:tag" content="Internal comms">
    <meta property="article:tag" content="Brand">
    <meta property="article:tag" content="Terra Carmichael">
    <meta property="article:tag" content="Eventbrite">
    
    <meta property="article:publisher" content="https://www.facebook.com/firstroundcapital/">
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="The Top Comms Mistakes Startups Make — And How To Avoid Them">
    <meta name="twitter:description" content="After two decades of comms experience with companies like Eventbrite, Yahoo, Mattel and Nike, Terra Carmichael shares four common PR mistakes and her techniques for sidestepping them.">
    <meta name="twitter:url" content="https://review.firstround.com/the-top-comms-mistakes-startups-make-and-how-to-avoid-them/">
    <meta name="twitter:image" content="https://review.firstround.com/content/images/size/w1200/2056/firstround-2f5nonhr6tspgdwcg5dj7v_vntr1912.jpg">
    <meta name="twitter:label1" content="Written by">
    <meta name="twitter:data1" content="First Round Staff">
    <meta name="twitter:label2" content="Filed under">
    <meta name="twitter:data2" content="PR &amp; Marketing, Sales, People &amp; Culture, Women, Content marketing, PR, Storytelling, Internal comms, Brand, Terra Carmichael, Eventbrite">
    <meta name="twitter:site" content="@firstround">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="800">
    
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
    "headline": "The Top Comms Mistakes Startups Make — And How To Avoid Them",
    "url": "https://review.firstround.com/the-top-comms-mistakes-startups-make-and-how-to-avoid-them/",
    "datePublished": "2018-07-20T22:54:10.000Z",
    "dateModified": "2024-03-06T12:42:11.000Z",
    "image": {
        "@type": "ImageObject",
        "url": "https://review.firstround.com/content/images/size/w1200/2056/firstround/5nonhr6tspgdwcg5dj7v_vntr1912.jpg",
        "width": 1200,
        "height": 800
    },
    "keywords": "PR & Marketing, Sales, People & Culture, Women, Content marketing, PR, Storytelling, Internal comms, Brand, Terra Carmichael, Eventbrite",
    "description": "After two decades of comms experience with companies like Eventbrite, Yahoo, Mattel and Nike, Terra Carmichael shares four common PR mistakes and her techniques for sidestepping them.",
    "mainEntityOfPage": "https://review.firstround.com/the-top-comms-mistakes-startups-make-and-how-to-avoid-them/"
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
  <body class="post-template tag-pr-and-marketing tag-sales tag-people-and-culture tag-women tag-content-marketing tag-pr tag-storytelling tag-internal-comms tag-brand tag-terra-carmichael tag-eventbrite tag-hash-wp tag-hash-wp-post tag-hash-import-2023-12-22-15-59 relative overflow-x-hidden bg-backgr text-typogr scroll-smooth "
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
                <h1 data-label="Article Title" class="text-[40px] lg:text-5xl tracking-tight !font-skandia-medium mt-5 lg:mt-8 leading-[1.15] mb-3">The Top Comms Mistakes Startups Make — And How To Avoid Them</h1>
                <p data-label="Description" class="text-2xl lg:text-3xl font-leif tracking-tight">After two decades of comms experience with companies like Eventbrite, Yahoo, Mattel and Nike, Terra Carmichael shares four common PR mistakes and her techniques for sidestepping them.</p>

            </div>
        </div>
    </div>

          <figure class="aspect-auto max-w-full min-lg:max-w-[1200px] max-lg:order-3 m-auto flex-1 lg:aspect-16/9!">
            <picture  >
  <source 
    srcset="/content/images/size/w320/format/webp/2056/firstround/5nonhr6tspgdwcg5dj7v_vntr1912.jpg 320w, /content/images/size/w640/format/webp/2056/firstround/5nonhr6tspgdwcg5dj7v_vntr1912.jpg 600w, /content/images/size/w960/format/webp/2056/firstround/5nonhr6tspgdwcg5dj7v_vntr1912.jpg 960w, /content/images/size/w1280/format/webp/2056/firstround/5nonhr6tspgdwcg5dj7v_vntr1912.jpg 1200w, /content/images/size/w2000/format/webp/2056/firstround/5nonhr6tspgdwcg5dj7v_vntr1912.jpg 2000w"
    sizes="100vw"
    type="image/webp"
  >
  <img class="w-full object-cover object-center lg:mt-0"
    
    loading="eager"
    srcset="/content/images/size/w320/format/webp/2056/firstround/5nonhr6tspgdwcg5dj7v_vntr1912.jpg 320w, /content/images/size/w640/format/webp/2056/firstround/5nonhr6tspgdwcg5dj7v_vntr1912.jpg 600w, /content/images/size/w960/format/webp/2056/firstround/5nonhr6tspgdwcg5dj7v_vntr1912.jpg 960w, /content/images/size/w1280/format/webp/2056/firstround/5nonhr6tspgdwcg5dj7v_vntr1912.jpg 1200w, /content/images/size/w2000/format/webp/2056/firstround/5nonhr6tspgdwcg5dj7v_vntr1912.jpg 2000w"
    sizes="100vw"
    src="/content/images/size/w30/2056/firstround/5nonhr6tspgdwcg5dj7v_vntr1912.jpg"
    alt="The Top Comms Mistakes Startups Make — And How To Avoid Them"
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
            href="https://twitter.com/share?text=The%20Top%20Comms%20Mistakes%20Startups%20Make%20%E2%80%94%20And%20How%20To%20Avoid%20Them&amp;url=https://review.firstround.com/the-top-comms-mistakes-startups-make-and-how-to-avoid-them/"
            title="Share on Twitter" aria-label="Tweet">
          <i class="icon icon-twitter-x stroke-1.5">
  <svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-brand-x hover:contrast-90 hover:brightness-110" width="16" height="16" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
  <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
  <path d="M4 4l11.733 16h4.267l-11.733 -16z"></path>
  <path d="M4 20l6.768 -6.768m2.46 -2.46l6.772 -6.772"></path>
</svg>
</i>          </a>
          <a class="flex flex-row justify-center items-center p-3 ibtn rounded-full transition duration-200 outline-hidden focus:outline-hidden hover:outline-hidden active:outline-hidden border border-black/25 h-[40px] w-[40px] text-black"target="_blank"
            href="https://www.linkedin.com/shareArticle?mini=true&url=https://review.firstround.com/the-top-comms-mistakes-startups-make-and-how-to-avoid-them/&title=The%20Top%20Comms%20Mistakes%20Startups%20Make%20%E2%80%94%20And%20How%20To%20Avoid%20Them&summary=The%20Top%20Comms%20Mistakes%20Startups%20Make%20%E2%80%94%20And%20How%20To%20Avoid%20Them"
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

        <p>When it comes to public relations, startups are often focused on what they can do to win coverage and secure that big story to help make a splash in a noisy space. So much energy is poured into landing the interview or getting that speaking gig, that it’s easy to under-index on the part that comes next: what you’re actually going to say.</p><p><strong><a href="https://www.linkedin.com/in/terracarmichael?ref=review.firstround.com">Terra Carmichael</a></strong>’s decades of experience in communications have shown her that from the smallest startups and greenest founders to the largest companies and most seasoned executives, most PR blunders are actually strikingly similar — and completely avoidable. Instead of chasing those glowing puff pieces or coveted cover stories, the current VP of Communications at <strong><a href="https://www.eventbrite.com/?ref=review.firstround.com">Eventbrite</a></strong> argues that leaders should be putting more thought into what <em>not</em> to do.</p><p>Because no matter what you're building, following Carmichael’s <a href="https://review.firstround.com/how-new-startups-can-win-at-pr-advice-from-a-20-year-comms-career/">expert advice on finding and pitching the right journalists</a> is only half the battle. You still need a game plan for navigating the interview itself. And from botched messages to those embarrassing yet unintentional gaffes, there are all too many ways for an interview to go off the rails.</p><p>Luckily, after working with companies such as Yahoo, Mattel, Nike and Trulia, Carmichael has a good sense of what to look out for and avoid. At First Round’s Founders Summit, she boiled it down to the four most common comms mistakes companies make, sharing her practical tips and techniques for sidestepping them.</p><h2 id="mistake-1-faking-it">MISTAKE #1: FAKING IT</h2><p>“Fake it till you make it” has long been a popular operating philosophy, especially among first-time founders or early startup teams finding their way to traction. But the truth has never been more important. And faking it publicly can have dire consequences.</p><p>“It’s not hard to see why this phrase has staying power,” says Carmichael. “There’s seemingly a lot of benefits to ‘faking it’. As a founder or executive, everyone seems to be watching: your team, your investors, the public. There’s a lot of pressure out there, whether it’s raising money, generating buzz or proving yourself as a leader. You’re expected to be inspirational and visionary, even when you don’t always feel that way or when the product isn’t quite there yet.”</p><p>So while faking it may seem like a great shortcut to willing things into existence, there are some real dangers lurking. “It’s easy to stretch it too far, flying too close to the sun while you’re waiting for reality to catch up to the hype. And faking it in a public setting carries some significant risks,” says Carmichael. “The very obvious and sad example of this is Theranos. It was a seemingly wildly successful startup, getting enviable press and at one point being valued at $10 billion. Now, in one of the most rapid, public meltdowns we’ve seen in startup land, the company is teetering on the edge of bankruptcy and executives are facing jail time.”</p><p>But faking it needn't rise to the level of criminality to become a problem. “There are loads of subtle and innocuous ways that people fake it and get carried away,” Carmichael observes.</p><p>Here are two of Carmichael’s tips to avoid getting swept away in exaggeration:</p><p><strong>Connect it back.</strong> Stay grounded in your company’s mission, vision, and values. “This may seem like big company speak, but it doesn’t have to be a daunting exercise. It can just be a few lines that can help center you and tie everything together. What it’s really about is being true to what your company stands for, and what your team is passionate about,” says Carmichael. “For example, if one of your values is centered around teamwork and collaboration, then you would work that into everything you do, whether it’s in evaluating future team members, rewarding behavior at work, making business decisions or talking about your achievements to the press.”</p><p><strong>Be deliberate.</strong> As an executive or spokesperson, you want to be intentional about <em>how</em> you speak and <em>what</em> you speak about. “For me, it really comes down to one question: <strong>Who are you to your core?</strong> Think about topics that you can authentically speak to with passion instead of speaking about things that you <em>wish</em> you were passionate about. You also should think about your style — when do you show up at your best? It’s possible that showing up at your best isn’t necessarily how you wished you came across,” says Carmichael. “For example, not all CEOs are funny, but many try to be when they are on stage because they think that’s what people want. The problem is that it ends up looking inauthentic and is often painful to watch. The reality is that when you are speaking, people want you to succeed. No one likes to see failure (at least not decent people), so show up as your true self, and you’ll win.”</p><p>For help with that, there are a couple of exercises Carmichael recommends. One is just a simple <strong>audit</strong>. “Ask your team or trusted colleagues to give you real feedback on where you crush it, and where you fall down. You’d be surprised what people will share with you when you simply ask and give them permission to tell you when things aren’t working,” she says.</p><p>Another way to achieve this is to develop an <strong>executive brand profile</strong> for key spokespeople. Carmichael worked with <a href="https://www.linkedin.com/in/helenamaus/?ref=review.firstround.com">Helena Maus</a> of <a href="http://www.biteglobal.com/?ref=review.firstround.com">Bite Communications</a> to build one for an executive she was advising, using an interview to determine where the leader’s authentic self could be brought in. Afterwards, they came up with three to four topics or platforms that were very clear, deliberate and clearly mapped back to the business. “These defined themes help focus you and ensure greater success. It’s like the squirt gun game at state fairs — if you point them all at the same target and focus your efforts, you’ll get there faster,” Carmichael explains. “Once the profile has been created, you can work from there to start seeking press, social engagement or speaking opportunities to advance your public persona and brand around these specific topics. But more importantly, this exercise is also a tool to filter out opportunities that <em>don’t</em> further your work or benefit your business. Not everything is worth your precious time.”</p><h2 id="mistake-2-oversharing">MISTAKE #2: OVERSHARING</h2><p>When you’ve discovered your authentic self, it’s easy to feel ready for primetime and eager to share your message with the world. But as Carmichael cautions, there is a hidden danger here: Saying too much.</p><p>According to Carmichael, there are many factors that lead people down the path of talking too much or speaking too transparently in interviews: sometimes it’s ego, the fear of being misunderstood, nerves or just wanting people to like you<strong>.</strong> Whatever the reason, it’s easy to slip up and slide into sharing too much. And it’s not always a matter of getting carried away or being too chatty, but rather sitting down for interviews that you shouldn’t be doing in the first place.</p><p>“Some folks have an intense ambition to get their message or their side of the story out there. But that can cloud your judgement when it comes to leveraging press,” says Carmichael. “For a non-tech and more sensational example, I always like to point to <a href="https://en.wikipedia.org/wiki/Mark_Sanford?ref=review.firstround.com">Mark Sanford</a>, the former governor of South Carolina. You might not remember, but he was caught in an affair with an Argentine woman, and it was this really big fiasco. But then Michael Jackson passed away unexpectedly. And that was just such a big moment, probably one of those few times in history when people remember where they were when they found out. And while tragic, from a PR perspective, for Sanford this really was an enormous opportunity to allow the news cycle on his affair to fade while mainstream media were consumed by something else. Jon Stewart even <a href="http://www.cc.com/video-clips/6aj2b5/the-daily-show-with-jon-stewart-shut-up--mark-sanford?ref=review.firstround.com">commented</a> on this topic, joking that ‘God killed Michael Jackson to save his ass.’ But instead Sanford jumped back in the spotlight and did a lengthy interview with the Associated Press, calling his mistress his soulmate. This is an example of someone who felt misunderstood and had a blinding need to get his version of the story out there. Obviously, this backfired. Going silent wouldn’t have changed the outcome for him, but the additional attention on the issue did him no favors.”</p><p>For Carmichael, it all comes down to knowing when to stop talking. Here’s what you need to watch out for to make sure you’re not sharing unnecessarily:</p><p><strong>Don’t fill up the silence.</strong> “One of the oldest tricks in the book that trained reporters use is silence,” says Carmichael. “You'll finish your answer or message, and they'll just sit and look at you, baiting you into saying more. Or they’ll quietly write in their notebooks, with their silence swallowing the room. And executives often feel this need to fill it right? It gets awkward and uncomfortable. So they just keep talking, digging themselves in further and further, creating more opportunities for a reporter to pounce on something that they really didn’t even mean to say. That’s because when you’ve run out of messaging and are just winging it, you often end up sharing more than you intended to, in a less than polished way.”</p><blockquote>Know your message, deliver it and then put a period on it.</blockquote><p><strong>Remember your role. “</strong>It’s important to remember that while it’s easy to be friendly with reporters (after all, most are very nice people), you have a job to do — and so do they,” Carmichael cautions. “It helps to remember your different roles. While not adversarial necessarily, you aren’t on the same team. Most journalists aren’t out to get you, but they certainly aren’t an extension of your comms team. They're not there to do your PR for you, so don't expect that. Their job is to find a story, and sometimes what you want to tell and what they want to hear don't match up. Essentially, they are looking for the juice and the headline, especially in today’s clickbait driven world. So that might not align to your goals, and it’s important to be mindful of that and make smart decisions about what to participate in and what to pass on.”</p><p><strong>Define the rules of engagement:</strong> As any trained comms person knows, there are certain terms you can use to navigate the conversation with reporters to maximize the chances of getting what you want. For example, you can frame a conversation as “off the record,” which means you can can influence a story by providing context without having your fingerprints on the story, if that is the best course. There are other handshake agreements that you can use with reporters when you already have a relationship with them, such as giving information “on background” or “exclusively” that can help you both achieve your goals.</p><p><strong>Stick to the truth.</strong> Though the truth may not be as sexy as reality, Carmichael says it’s absolutely the way to go. “Don’t embellish or stretch the truth beyond recognition because you can always count on reporters to sniff out the facts, as they should. Beyond the ethical case for telling the truth, it’s also easier to remember,” she points out. “Sometimes startups are tempted to stretch too far when describing the nature of partnerships or integrations in order to leverage a larger brand or showcase momentum. But in addition to running the risk of being untruthful, you also could fall out of the good graces of existing or potential partners that could be useful to you in the future, so it’s a dangerous move.”</p><figure class="kg-card kg-image-card kg-card-hascaption"><img src="https://review.firstround.com/content/images/2056/firstround/yyohjr7hqguutbrvvgfw_vntr1951.jpg" class="kg-image" alt loading="lazy"><figcaption>Eventbrite VP of Communications Terra Carmichael</figcaption></figure><h2 id="mistake-3-answering-the-questions">MISTAKE #3: ANSWERING THE QUESTIONS</h2><p>With interviews, most worry about the questions they’ll get asked and how they’ll get tripped up. But according to Carmichael, you have more control over this than you may think. “When it comes to the press, you may not have much control on the final draft or on how the story turns out, but you have complete control of <em>what</em> you say and <em>how</em> you say it,” says Carmichael. “I always tell the executives I’m working with that navigating interviews is an art — there’s a way to master it, but it takes work.”</p><blockquote>The goal of an interview isn’t to answer questions. It’s to deliver your messages.</blockquote><p>Here are four techniques Carmichael recommends to guide the conversation where you want it to go:</p><p><strong>Limit yourself to your top messages.</strong> When you land a new interview opportunity, immediately start thinking about the top three messages you want to get across. “You don’t want to have so many topics that you’re trying to cram in a single interview,” Carmichael says. “That’s just too much to remember and you’ll get caught up in trying to deliver what you’ve memorized or in worrying about leaving one point out, which will make you seem distracted or inauthentic. What it’s really about is whittling down what you want to say, distilling your company’s narrative or announcement into a few key points that you can anchor your answers on. And then under that you can fill in the color.”</p><p><strong>Prepare for the tough ones.</strong> Of course, focusing on getting your points across doesn’t mean that you shouldn’t give any thought to what a reporter might ask. Do your homework by reading up on the journalist’s recent pieces to get a feel for his or her style. <a href="https://review.firstround.com/Powerful-Tips-from-Techs-Top-Media-Trainer-and-Speaking-Coach/">Start by making a list of the questions you fear most</a>. That way you can head into any interview knowing the five worst things you could be asked and have at least an outline of how you’d tackle them. “These can include scrutinizing questions around your business model, competitive landscape, exit plans and timing. It’s also important to be real about any potentials skeletons that may be hiding in your closet, and prepare as if they are going to be revealed,” Carmichael advises.</p><p><strong>Block and bridge.</strong> A go-to technique many successful spokespeople use for navigating interviews is what the industry calls blocking and bridging<strong>.</strong> “In a nutshell, you <em>block</em> by acknowledging and briefly answering the question that's been asked, and then you <em>bridge</em> to the message you actually want to deliver. Think of it as jumping off point to the information you want to shine a spotlight on,” Carmichael explains. “Once you have your top three messages outlined, you should see if can take any question and bring it back to your messaging.”</p><p>Here are two block-and-bridge starter statements to have in your back pocket:</p><ul><li>“That’s an interesting observation...but the heart of the matter is really…”</li><li>“You raise a good point about a key issue in our industry today…How we like think about XYZ is…”</li></ul><p><strong>Avoid the negative.</strong> There are a number of techniques that reporters use to try to get you to go down a dark path or give a negative answer that makes a good headline, but isn’t good for your business.  “If you're ever asked a loaded question or something framed as a negative statement, never repeat the negative. Doing so ends up putting you on your heels with connectives such as ‘but’ or ‘however.’ Using words like that make you sound defensive,” says Carmichael. “And it's just not worth it. Don’t play back the negative sentiment. Start fresh with your perspective.” Here are three statements you can use to redirect and take the conversation where you want to go could be:</p><ul><li>“Another way of thinking about this is…”</li><li>“I think what you’re really asking is…”</li><li>“That speaks to a bigger point about…”</li></ul><p>The same advice applies when you’re asked about a problem your company or industry is facing. Don’t reiterate the problem, but instead talk about a solution. As an example, Carmichael cites an interview where a <a href="https://www.youtube.com/watch?v=HNeaivMa7pA&ref=review.firstround.com">Lenovo executive</a> did a masterful job of of bridging as he’s faced with probing questions about their earnings on the air.</p><p><strong>Don’t be intentionally obtuse.</strong> While the goal may be to deliver messages instead of answering questions, you don’t want to come across as intentionally frustrating or evasive. It can’t be purely about pushing your own agenda and you don’t want to be too transparent or obvious in your pivots. “You certainly don’t want to create one of those viral clips where the interviewee or politician doggedly refuses to answer the question and gets called out on it,” says Carmichael. “Just like in everyday relationships, there’s a give and take that you need to balance in a way that is respectful and amicable.”</p><h2 id="mistake-4-forgetting-your-number-one">MISTAKE #4: FORGETTING YOUR NUMBER ONE</h2><p>When it comes to strategic communications, it’s very easy to get wrapped up in dealing with reporters or crafting your external message. But in Carmichael’s experience, another major mistake leaders and companies often make is that they forget that their most important audience is already inside their building.</p><p>“I’ve long believed that internal comms is often overlooked,” says Carmichael. “Too many executives either don't make the time, or think about it as an afterthought or nice-to-have. But in my view, employees really should be your number one audience, especially as you scale. It’s absolutely critical that you communicate with them authentically and often."</p><p>What’s more is that this is a captive, highly invested group that can actually help your external PR efforts. <strong>When testing your message or story, make sure it resonates on the inside first.</strong> “Your employees are often your toughest critics,” says Carmichael. “They will call bullshit on you faster than anyone else. So if your story is not resonating with them, it's probably not going to resonate on the outside. Use them, and tap into the vested interest that they have in your company to help you tell your story better.”</p><p>But workshopping PR messaging isn’t the main reason to communicate inside. The industry is peppered with studies and data around the importance of employee engagement, highlighting that perks and benefits don’t necessarily mean your employees are engaged or believe in your mission. Rather, a culture that is built on accountability, authenticity, clear goals and strong two-way communication is what goes a long way to keeping your team engaged.</p><p>Carmichael recently invested in dedicated internal comms resources at Eventbrite and the team has been trying new ways to communicate to employees (lovingly called Britelings) more frequently — all based on a few tried and true learnings. Here’s how she’s going deeper on internal comms:</p><ul><li><strong>Leave it to the pros</strong>. “After five years at Eventbrite, I finally hired someone who knew much more than I did and specialized in internal and employee comms,” says Carmichael. “We’ve seen a big difference, especially among leaders and mid-level managers who appreciate the efforts to help cascade communications throughout the organization.”</li><li><strong>Read all about it.</strong> “I’m a big believer in teaching leaders to fish. That’s why we’ve rolled out a weekly(ish) email for leaders at Eventbrite that summarizes all the things they need to be thinking about in terms of managing and messaging to their team. We break it down into a few sections: things to know, things to do, things to share,” says Carmichael. “It sounds simple, but let’s be real, leaders are an important and expensive investment, so you want to use their time wisely. Giving them the tools such as cliff notes and summaries of important matters not only helps them be better communicators, but it also enables a timely cascade of communications throughout the organization.”</li><li><strong>Find micro-moments to connect.</strong> Aside from company-wide touchpoints, it’s also important to create moments to connect in small groups. “We've recently started a weekly coffee break with our CEO, Julia Hartz, where six people get to sign up to informally discuss a specific topic with her,” says Carmichael. “Not only does this give employees valuable face time with her, but it's also been fantastic for Julia because it’s a new way for her to talk to a broader cross-section of the team and get their thoughts on topics that are top-of-mind for her. She’s also able to get a pulse on how people of all levels of the organization are feeling and any concerns or issues that they bring up themselves.”</li></ul><blockquote>Just doing an all-hands or a town hall meeting once a month doesn't cut it anymore. Employees need to feel a deeper connection to you and the business.</blockquote><h2 id="above-all-put-in-the-work">ABOVE ALL, PUT IN THE WORK</h2><p>Whether you’re bringing in a comms leader or you’re a founder of a small startup that <a href="https://review.firstround.com/why-most-startups-dont-get-press/">wants to go it alone</a>, the bottom line is that if you’ve put in the time cultivating your media network and getting in front of reporters, you should invest even more effort in learning how to talk to them and how to get your message across. Success comes down to understanding that, while communications and press can drive a business forward, they really need to be managed well. Navigate these testy waters by grounding yourself in your authentic truth, being laser focused on your story and your message, and most of all, by prioritizing your internal team.</p><p>“A lot of founders and executives at companies think that they've got their story nailed because they have mastered their pitch with investors, employees or even their parents and friends, but the press is a totally different animal,” says Carmichael. “You’ve got to be humble and put in the work, even if you don’t think you need it. You’re likely not as good as you think you are, and there’s always room for improvement.”</p>
      </article>

      <div class="flex flex-row justify-center items-center border-t border-black/20 pt-8 pb-8 max-w-content m-auto w-full max-lg:px-5 gap-x-4">
        <span class="text-sm text-black">Share on:</span>
          <div class="flex flex-row justify-center md:flex xl:col-span-8" data-share>
    <div data-label="Share Article" class="flex flex-row space-y-0 sm:space-y-12 items-center">
      <div class="flex flex-row items-center space-x-2 sm:space-x-2">
        <div data-label="Share Options" class="flex flex-row space-x-2 sm:space-x-2">
          <a class="flex flex-row justify-center items-center p-3 ibtn transition duration-200 rounded-full outline-hidden focus:outline-hidden hover:outline-hidden active:outline-hidden border border-black/25 h-[40px] w-[40px] text-black"  target="_blank"
            href="https://twitter.com/share?text=The%20Top%20Comms%20Mistakes%20Startups%20Make%20%E2%80%94%20And%20How%20To%20Avoid%20Them&amp;url=https://review.firstround.com/the-top-comms-mistakes-startups-make-and-how-to-avoid-them/"
            title="Share on Twitter" aria-label="Tweet">
          <i class="icon icon-twitter-x stroke-1.5">
  <svg xmlns="http://www.w3.org/2000/svg" class="icon icon-tabler icon-tabler-brand-x hover:contrast-90 hover:brightness-110" width="16" height="16" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" fill="none" stroke-linecap="round" stroke-linejoin="round">
  <path stroke="none" d="M0 0h24v24H0z" fill="none"></path>
  <path d="M4 4l11.733 16h4.267l-11.733 -16z"></path>
  <path d="M4 20l6.768 -6.768m2.46 -2.46l6.772 -6.772"></path>
</svg>
</i>          </a>
          <a class="flex flex-row justify-center items-center p-3 ibtn rounded-full transition duration-200 outline-hidden focus:outline-hidden hover:outline-hidden active:outline-hidden border border-black/25 h-[40px] w-[40px] text-black"target="_blank"
            href="https://www.linkedin.com/shareArticle?mini=true&url=https://review.firstround.com/the-top-comms-mistakes-startups-make-and-how-to-avoid-them/&title=The%20Top%20Comms%20Mistakes%20Startups%20Make%20%E2%80%94%20And%20How%20To%20Avoid%20Them&summary=The%20Top%20Comms%20Mistakes%20Startups%20Make%20%E2%80%94%20And%20How%20To%20Avoid%20Them"
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
          id: "6585b26b580f2439a1e0ba86",
          tags: ["PR &amp; Marketing", "Sales", "People &amp; Culture", "Women", "Content marketing", "PR", "Storytelling", "Internal comms", "Brand", "Terra Carmichael", "Eventbrite", "#wp", "#wp-post", "#Import 2023-12-22 15:59"],
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
