<!-- URL: https://partnermarketinghub.withgoogle.com/brands/google-news/use-cases/press-and-media/ -->
<!-- HTTP: 200 SIZE:   333669 -->






<!DOCTYPE html>
<html lang="en-us"
      dir="ltr">

    <head>
        <meta charset="utf-8" />
        <title>
            
                Press and media
            
            
                
                    
                        - Partner Marketing Hub
                    
                
            
        </title>
        
            <meta name="description"
                  content="Partner Marketing Hub is the go-to resource to help bring your partnership with Google to life – brand guidelines, messaging opportunities, tools, and more." />
        
        <meta name="viewport" content="width=device-width, initial-scale=1" />

    
        <script>
            const hash = window.location.hash;
            if (hash && hash.startsWith('#/')) {
                const file = document.createElement('script');
                file.setAttribute('async', '');
                file.setAttribute('src', 'https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/js/redirects.js');
                document.head.appendChild(file);
            }
        </script>

        <link rel="dns-prefetch" href="https://fonts.googleapis.com" />
        <link rel="dns-prefetch" href="https://ajax.googleapis.com" />
        <link rel="dns-prefetch" href="https://www.gstatic.com" />

        <link rel="preconnect" href="https://storage.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" />

    <!--Page info-->
        <script>
            window.pmh = {};
            window.pmh.locale = 'en-us';
            window.pmh.brandName = 'Google News';
            
                window.pmh.pageId = parseInt('775');
                window.pmh.canonicalPage = parseInt('');
            
        </script>

    <!--Analytics container-->
        
        
            

<link rel="dns-prefetch" href="https://stats.g.doubleclick.net" />
<link rel="dns-prefetch" href="https://www.google-analytics.com" crossorigin />
<link rel="dns-prefetch" href="https://www.googletagmanager.com" crossorigin />




<!-- Google Tag Manager -->
<script>
  function glueCookieNotificationBarLoaded() {
    const {
      instance
    } = window.glue.CookieNotificationBar;
    if (!instance) return;

    if (!window.pmh || !window.pmh.user) {
      getUserData('/backend/api/user/');
    }
  }

  async function getUserData(endpoint) {
    const response = await fetch(endpoint);
    const data = await response.json();
    setUser(data);
    setInitializeEvent();
    pushMainData();
    initAnalyticsContainer();
  }

    // GA4: Need to fire this event on each page load immediately before GTM loads on the page.
  function setInitializeEvent() {
    window.dataLayer = window.dataLayer || []
    window.dataLayer.push({
      event: 'dataLayer_initialized',
      page_name: document.title,
      experiments: undefined, // PMH currently not using any experiments, so no value assigned.
      is_google_user: window.pmh.user.isGoogleUser,
      user_state: window.pmh.user.approvalStatus || 'not logged in',
      brand_name: window.pmh.brandName
    });
  }

  function initAnalyticsContainer() {
    installAnalyticsContainer(
      window,
      document,
      'script',
      'dataLayer',
      'GTM-MGK94WK',
      true /* trackAllClicks */ ,
      false /* isSpa (single page app) */ );
  }

  function installAnalyticsContainer(w, d, s, l, i, trackAllClicks, isSpa) {
    w[l] = w[l] || [];
    w[l].push({
      'gtm.start': new Date().getTime(),
      event: 'gtm.js',
      isSpa: Boolean(isSpa),
      trackAllClicks: Boolean(trackAllClicks)
    });
    var f = d.getElementsByTagName(s)[0];
    var j = d.createElement(s);
    var dl = l != 'dataLayer' ? '&l=' + l : '';
    j.async = true;
    j.src = 'https://www.googletagmanager.com/gtm.js?id=' + i + dl;
    f.parentNode.insertBefore(j, f);
  }

  function setUser(data) {
    if (!window.pmh) {
      window.pmh = {};
    }

    window.pmh.user = data.user;

    if (window.pmh.user) {
      window.pmh.user.approvalStatus = data.user.approval_status;
      window.pmh.user.isAuditor = data.user.is_auditor || false;
      window.pmh.user.isGoogleUser = data.user.is_google_user || false;
      window.pmh.user.needsReview = data.user.needs_review;
      window.pmh.user.shouldCompleteApplication = data.user.should_complete_application || false;
      window.pmh.user.topPartner = data.user.top_partner || false;
    }
  }

  function pushMainData(user) {
    const dataObj = {
      isGoogleUser: window.pmh.user.isGoogleUser || false,
      pageLanguageCode: 'en-us' || window.pmh.locale,
      topPartner: window.pmh.user.topPartner ? 1 : 0,
      userState: getUserStatus(user),
    };
    const path = window.location.href;
    const isAccessForm = path.includes('request-access-form');

    if (isAccessForm) {
      dataObj['event'] = 'trackPageview';
      dataObj['virtualPageview'] = window.location.href;
    }

    window.dataLayer = window.dataLayer || [];
    window.dataLayer.push(dataObj);
  }

  function getUserStatus() {
    if (window.pmh.user.approvalStatus) {
      return window.pmh.user.approvalStatus
        .split(/\s+/)?.[0]
        .replace(/\(/g, '')
        .replace(/\)/g, '')
        .replace(/,/g, '')
        .replace(/'/g, '')
    }

    return 'not logged in';
  }
</script>
<!-- End Google Tag Manager -->

        

    <!--Open Graph Metadata-->
        
            
            <meta property="og:image"
                  content="https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/images/og_image.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&amp;X-Goog-Credential=wagtail%40pe-portal-consumer-prod.iam.gserviceaccount.com%2F20260510%2Fauto%2Fstorage%2Fgoog4_request&amp;X-Goog-Date=20260510T101705Z&amp;X-Goog-Expires=86400&amp;X-Goog-SignedHeaders=host&amp;X-Goog-Signature=2470410809503cfcaf03de362f4d271a9b735c4470de2e9a17c282b228b9f2146fe093c4c70307244c40a04f77afd5958ba7f92c1a336062ef868d40ecf73c9be0220bfc63e404c4228c01d8f072c49ecad9db4ba79f5ad0ac6959b46b5764f032cef98c2083c01aa84c02f48112636dd47f9baef4e1949d286bca4bd561f595d6fbdd6acef4e42393a21ac4baa24f0638ab629a2b135592eeaa7b292ee3f277097c19b45f220078994ff1ce479d94548fff6dd438be0ac86268613de2538ffcefb5df3a42328ab78de00b81a970272b7c59d624c7c627cc4affca98765dfe4b848ef31115fc375c0bbc8f0355a4774c1dd5bd12ab628db40b93f0bdee838708" />
            <meta property="og:title"
                  content="Press and media - Partner Marketing Hub" />
            
                <meta property="og:description"
                      content="Partner Marketing Hub is the go-to resource to help bring your partnership with Google to life – brand guidelines, messaging opportunities, tools, and more." />
            
            <meta property="og:site_name" content="Google" />
            <meta property="og:type" content="website" />
            <meta property="og:url" content="https://partnermarketinghub.withgoogle.com/brands/google-news/use-cases/press-and-media/" />
        

    <!--Twitter Card Metadata-->
        
            <meta name="twitter:card" content="summary" />
            <meta name="twitter:site" content="@google" />
        

        <meta name="theme-color" content="#ffffff" />

        <link rel="search" type="application/opensearchdescription+xml" href="https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/opensearch.xml?X-Goog-Algorithm=GOOG4-RSA-SHA256&amp;X-Goog-Credential=wagtail%40pe-portal-consumer-prod.iam.gserviceaccount.com%2F20260510%2Fauto%2Fstorage%2Fgoog4_request&amp;X-Goog-Date=20260510T101705Z&amp;X-Goog-Expires=86400&amp;X-Goog-SignedHeaders=host&amp;X-Goog-Signature=4f60f59c31f90d9f74f52477dd4144d25bd2698c0695e4e18b0ca2d913146215c9b9a2c8bd7f24a8c9e7d60098740a38cb3f7d70a2567e7555f0ae11f1bb527b5fe252e69b7782fcf7925a4c573934ca03fab8c62e59851ca267491c92217a134b32f3a7909678fa7c495c1745872c3a507b2ff425805eb20354946722752ff38181b90028891495c340b405171d961c8b897d81bc5f9aa1fba3dc7a8e343f347156e60fe68d6b2d15788310cdfd0ce90cbab584c535f39bb3f448f37794dd76959b394ff36e352759144484b3bf422c62c9863cac9deea375c6bdbd8f4b76531e45d5fb2036e07075549ae8fa311fb8f6dbf35a3eabcae8054f1be93cb5b843"
              title="Partner Marketing Hub" />

        <link rel="shortcut icon" href="https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/images/favicon.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&amp;X-Goog-Credential=wagtail%40pe-portal-consumer-prod.iam.gserviceaccount.com%2F20260510%2Fauto%2Fstorage%2Fgoog4_request&amp;X-Goog-Date=20260510T101705Z&amp;X-Goog-Expires=86400&amp;X-Goog-SignedHeaders=host&amp;X-Goog-Signature=56cf84358f3e3d8f683076a5d6b1ce01c1bca795c97be5da0ed629f4694faa909e01226d4092953c9a1ef57c480fd1bb6fee4c24da82a86600f4d8d67a65d316d9abf72108ab4d0d6f9b895abef88a11901606914f986cf3c68683596c7ce929be4d38a75fb4d81b028d9a56dadebc75dd92947360c925cd792d67afddcfb1f0259bba06eba86d6ff6679927ad4875543f8397cf6ba97c32fb515266997de304c67e9297c3a11120c93b90218e95bf52a5a32c368bf58b734affd36bf807e2145a5c2a538367f50344a856f13ac81ebb5c0852611f9731b11018ccfa44e3c64f4cd93e1ef544e7827e976e90629db7833b34a2a9d2b8248f61987cf4b3b59c6b" type="image/png" />

        <link rel="apple-touch-icon" href="https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/images/partner-marketing-hub.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&amp;X-Goog-Credential=wagtail%40pe-portal-consumer-prod.iam.gserviceaccount.com%2F20260510%2Fauto%2Fstorage%2Fgoog4_request&amp;X-Goog-Date=20260510T101705Z&amp;X-Goog-Expires=86400&amp;X-Goog-SignedHeaders=host&amp;X-Goog-Signature=198c36e620c52a50c3ba16bacf0dbcb56f84f32d94864554d147e5baf61d6736fc2209e045c9f1da30c22398087547c4a22060ab4e2beae5baecbbb4b7911dc5645a2c7e322785731785d2baa98e7de5c487ab389118fe56283b21eb5b4e507ef312d27f6b54d42ec69b2ef2b09c6dc15971ce405b13f0e5e78ac98855692a6dcd4fe86496b35d3c2661bdaa575dbd910f85fb255f329c36b0cd463e14f1fc469706c2c30a73e96f8476b79aca441f04b159b09dfcbe01c3edeaaf13a1d7ba80713605c36f1b31368f03c29b41a769e9ac98e3f1492fa769cd0b78451553b7518a1d03bc16f87e3407a93fe2ee5050cc6ba8593574d996fcd1ce591aefcb41e9" />

    <!--Global stylesheets-->
        
            <link rel="preload" as="style"
                  href="https://fonts.googleapis.com/css?family=Google+Sans+Text:400,500,700|Google+Sans+Display:400,500,600,700|Google+Sans:400,500,700|Product+Sans:400|Material+Icons+Outlined|Material+Icons&display=swap&lang=en"
                  onload="this.onload=null;this.rel='stylesheet'" />
            <noscript>
                <link rel="stylesheet"
                      href="https://fonts.googleapis.com/css?family=Google+Sans+Text:400,500,700|Google+Sans+Display:400,500,600,700|Google+Sans:400,500,700|Product+Sans:400|Material+Icons+Outlined|Material+Icons&display=swap&lang=en" />
            </noscript>
        

        <style>
            .pmh-hide {
                display: none !important;
            }

            .glue-header__bar--desktop,
            .subnav-brand__button {
                visibility: hidden;
            }
        </style>

        

<link href="https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/css/hubdesignsystem.fcbe103ec8.css" rel="preload" as="style" />
<link href="https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/css/hubdesignsystem.fcbe103ec8.css" rel="stylesheet" />


        
    

<link href="https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/css/brands.dd32940fe4.css" rel="preload" as="style" />
<link href="https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/css/brands.dd32940fe4.css" rel="stylesheet" />



        
            <link rel="preload" as="style" href="https://www.gstatic.com/glue/v25_0/glue-material.min.css" />
            <link rel="stylesheet" href="https://www.gstatic.com/glue/v25_0/glue-material.min.css" />
        

        <noscript>
            <link rel="stylesheet" href="https://www.gstatic.com/glue/v25_0/glue-no-js.min.css" />
        </noscript>

        <link rel="manifest" href="https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/site.webmanifest?X-Goog-Algorithm=GOOG4-RSA-SHA256&amp;X-Goog-Credential=wagtail%40pe-portal-consumer-prod.iam.gserviceaccount.com%2F20260510%2Fauto%2Fstorage%2Fgoog4_request&amp;X-Goog-Date=20260510T101705Z&amp;X-Goog-Expires=86400&amp;X-Goog-SignedHeaders=host&amp;X-Goog-Signature=8496211b0322f32cf4e36c8d79d7a45f1b628d66ba5f59cc156e6da7680d8e36bd4bf0fd89890086db9bd08e214fd0cecb644483f94a8c7473db3fb067fc26c842c04f039fe11509e21e8b8273d57605a5446d8c63977d9d4e9856f0ee0956dc665426ad6a43ccc9dd0f409c3d47f3dec8e8ea2d4277a3a237170b9dd086aad22dfa75d167fbd058ff168dd55fc0a79c0353ba91f1cffb237adcce2c3c8525707b4754077cc92c75ef49d2bb512bfc4b3ff25fde10c278562f7c5698a141386dc0879cdee598ef8d3841cd6561b113bce4d0ece7c84781a253ffa3a24e3ff631a49f6ca7b1a2190079ddbd3e5b4bdc14d538bbd14632b237496874f1afa93805" />

        
    
    
    

    <script type="application/ld+json">
    {
        "@context": "http://schema.org/",
        "@type": "Brand",
        
         "logo": https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/images/googlenews_breadcrumb.original.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&amp;X-Goog-Credential=wagtail%40pe-portal-consumer-prod.iam.gserviceaccount.com%2F20260510%2Fauto%2Fstorage%2Fgoog4_request&amp;X-Goog-Date=20260510T101705Z&amp;X-Goog-Expires=86400&amp;X-Goog-SignedHeaders=host&amp;X-Goog-Signature=4b7d63c88aa0ba78b8f3fd844fb0938e9c014c5807fa02c60b7a2c294ea968c685356fbf1e0ab7e5b1481ba8b5d671fe93de9d5d2d56f7604d991726644f19d943083fc4d854d680623d7ea517b701f535bdec69ae9c461247da4d7e79a7a7db93538b661e3e250e404f6066cb68e6f4338e4fc9e004aaec5372336faf1b3b42590f60ff42a600aa9ae23ef058c70c90bb6533c3ae55cf4f6a60f4ff216e448c5c49ac8a018ca242ce15f2c6d8a707d917c62052e377f7a526965d3bc83bb4bf66bf5ab01e61a3f08b1e773da83c50a991cab26b8bfe2f6c39343f3e04714c96b68c7a862b530fd2a6e78150b4cb05aa72e3221fbcc4aad76a55c23419b6d2c0,
        
        
    }
</script>


        
    
        
    </head>

    <body class="glue-body " id="page-top" data-component="body"
          data-ga4-title="Press and media" 
          data-id="775" >
        
    <!-- Google Tag Manager (noscript) -->
            <noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-MGK94WK" height="0" width="0"
                              style="display: none; visibility: hidden;"></iframe></noscript>
    <!-- End Google Tag Manager (noscript) -->
        

        
            


    <div data-component="audit-notifications" class="pmh-hide">
        <div class="audit-notifications-floating-nav">
            <button aria-controls="pmh-dialog-main" aria-haspopup="dialog" class="audit-notifications-floating-trigger glue-button
                glue-button--high-emphasis glue-button--icon" data-ga4-analytics-event="{
                    'event': 'content_audit_portal_open'
                    }">
                <svg aria-hidden="true" class="audit-notifications-floating-icon glue-icon">
                    <svg width="20" height="20" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M5.58 1.83L4.15.4C1.75 2.23.17 5.05.03 8.25h2a8.445 8.445 0 0 1 3.55-6.42zm12.39 6.42h2c-.15-3.2-1.73-6.02-4.12-7.85l-1.42 1.43a8.495 8.495 0 0 1 3.54 6.42zm-1.97.5c0-3.07-1.64-5.64-4.5-6.32v-.68c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v.68C5.63 3.11 4 5.67 4 8.75v5l-2 2v1h16v-1l-2-2v-5zm-6 11c.14 0 .27-.01.4-.04.65-.14 1.18-.58 1.44-1.18.1-.24.15-.5.15-.78h-4c.01 1.1.9 2 2.01 2z" fill="#fff"/></svg>
                </svg>
                <span class="audit-notifications-floating-help-text glue-visually-hidden">
                    View Audit actions
                </span>
            </button>
        </div>

        <div aria-labelledby="dialog-label-main" class="pmh-dialog pmh-hide" id="pmh-dialog-main" role="dialog"
            aria-hidden="true" data-component="dialog">
            <h2 id="dialog-label-main" class="glue-headline glue-headline--headline-4">
                Is this content up to date?
            </h2>
            <p>
                Your content is due for an audit — please review.
            </p>
            <div class="pmh-dialog__actions">
                <button class="glue-button glue-button--high-emphasis" aria-haspopup="dialog" data-ga4-analytics-event="{
                    'event': 'content_audit_portal_cta_click',
                    'link_text': 'Yes, this is up to date'
                    }">
                    Yes, this is up to date
                </button>
                <button class="glue-button glue-button--medium-emphasis" aria-haspopup="dialog" data-ga4-analytics-event="{
                    'event': 'content_audit_portal_cta_click',
                    'link_text': 'No, edit now'
                    }">
                    No, edit now
                </button>
                <button class="glue-button glue-button--low-emphasis" aria-haspopup="dialog" data-ga4-analytics-event="{
                    'event': 'content_audit_portal_cta_click',
                    'link_text': 'No, please unpublish'
                    }">
                    No, please unpublish
                </button>
            </div>
            <div class="pmh-dialog__footer">
                <p class="glue-small-text">
                    It's important to take action to ensure content is accurate and still valid.
                </p>
            </div>
            <button type="button" class="glue-modal__close-btn" aria-label="Close dialog"
                data-analytics-on-click="{
                    'eventAction': 'modal exit click',
                    'eventLabel': 'audit - exit',
                    'componentName': 'modal'
                    }" data-ga4-analytics-event="{
                        'event': 'content_audit_portal_close'
                        }"></button>
        </div>

        <div aria-labelledby="pmh-dialog-unpublish" class="pmh-dialog pmh-dialog--unpublish pmh-hide"
            id="pmh-dialog-unpublish" role="dialog" aria-hidden="true" data-component="dialog">
            <h2 id="dialog-label-unpublish" class="glue-headline glue-headline--headline-4">
                Does this page need to be removed?
            </h2>
            <p>
                Once the request is met, the content will no longer be visible.
            </p>
            <div class="pmh-dialog__actions">
                <button class="glue-button glue-button--high-emphasis"
                    data-label="You have requested to unpublish this page." data-ga4-analytics-event="{
                        'event': 'content_audit_portal_cta_click',
                        'link_text': 'Yes, please unpublish'
                        }">
                    Yes, please unpublish
                </button>
                <button class="glue-button glue-button--medium-emphasis" data-ga4-analytics-event="{
                    'event': 'content_audit_portal_cta_click',
                    'link_text': 'Cancel'
                    }">
                    Cancel
                </button>
            </div>
            <div class="pmh-dialog__footer">
                <p class="glue-small-text">
                    This request may take up to 24 hours to be removed from the site.
                </p>
            </div>
            <button type="button" class="glue-modal__close-btn pmh-dialog__back-btn" aria-label="Go back"
                data-analytics-on-click="{
                    'eventAction': 'modal back click',
                    'eventLabel': 'audit - back',
                    'componentName': 'modal'
                    }">
                <svg aria-hidden="true" class="audit-notifications-floating-icon glue-icon">
                    <svg width="16" height="16" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M16 7H3.83l5.59-5.59L8 0 0 8l8 8 1.41-1.41L3.83 9H16V7z" fill="#000"/></svg>

                </svg>
            </button>
        </div>

        <div aria-labelledby="dialog-label-edit" class="pmh-dialog pmh-dialog--edit pmh-hide" id="pmh-dialog-edit"
            role="dialog" aria-hidden="true" data-component="dialog">
            <h2 id="dialog-label-edit" class="glue-headline glue-headline--headline-4">
                Would you like to edit now?
            </h2>
            <div class="pmh-dialog__actions">
                <a class="glue-button glue-button--high-emphasis" href="/admin/pages/775/edit/"
                    data-ga4-analytics-event="{
                        'event': 'content_audit_portal_cta_click',
                        'link_text': 'Yes, edit now'
                        }">
                    Yes, edit now
                </a>
                <button class="glue-button glue-button--medium-emphasis" data-ga4-analytics-event="{
                    'event': 'content_audit_portal_cta_click',
                    'link_text': 'Cancel'
                    }">
                    Cancel
                </button>
                <button class="glue-button glue-button--low-emphasis"
                    data-label="You have added this page to your “in progress” queue.">
                    Add to in progress queue
                </button>
            </div>
            <div class="pmh-dialog__footer">
                <p class="glue-small-text">
                    You can choose to edit now and be taken directly to the Wagtail draft page, or you can add this page to your “in progress” queue and edit later.
                </p>
            </div>
            <button type="button" class="glue-modal__close-btn pmh-dialog__back-btn" aria-label="Go back"
                data-analytics-on-click="{
                    'eventAction': 'modal back click',
                    'eventLabel': 'audit - back',
                    'componentName': 'modal'
                    }">
                <svg aria-hidden="true" class="audit-notifications-floating-icon glue-icon">
                    <svg width="16" height="16" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M16 7H3.83l5.59-5.59L8 0 0 8l8 8 1.41-1.41L3.83 9H16V7z" fill="#000"/></svg>

                </svg>
            </button>
        </div>

        <div aria-labelledby="dialog-label-up-to-date" class="pmh-dialog pmh-dialog--uptodate pmh-hide"
            id="pmh-dialog-up-to-date" role="dialog" aria-hidden="true" data-component="dialog">
            <h2 id="dialog-label-up-to-date" class="glue-headline glue-headline--headline-4">
                Are you sure all of this content is accurate?
            </h2>
            <p>
                We will check back with you at a later date.
            </p>
            <div class="pmh-dialog__actions">
                <button class="glue-button glue-button--high-emphasis"
                    data-label="This page was marked as “up to date.”" data-ga4-analytics-event="{
                        'event': 'content_audit_portal_cta_click',
                        'link_text': 'Yes, this is up to date'
                        }">
                    Yes, this is up to date
                </button>
                <button class="glue-button glue-button--medium-emphasis" data-ga4-analytics-event="{
                    'event': 'content_audit_portal_cta_click',
                    'link_text': 'Cancel'
                    }">
                    Cancel
                </button>
            </div>
            <div class="pmh-dialog__footer">
                <p class="glue-small-text">
                    Moving forward with this option will reset the expiration date and you will not be asked to audit this page again until it expires.
                </p>
            </div>
            <button type="button" class="glue-modal__close-btn pmh-dialog__back-btn" aria-label="Go back"
                data-analytics-on-click="{
                    'eventAction': 'modal back click',
                    'eventLabel': 'audit - back',
                    'componentName': 'modal'
                    }">
                <svg aria-hidden="true" class="audit-notifications-floating-icon glue-icon">
                    <svg width="16" height="16" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M16 7H3.83l5.59-5.59L8 0 0 8l8 8 1.41-1.41L3.83 9H16V7z" fill="#000"/></svg>

                </svg>
            </button>
        </div>
    </div>


        

        
            
        

        
            



    <header
        class="glue-header glue-header--single pmh-header "
        data-component="header">
        <div class="complete-application pmh-hide">
            <p class="glue-body">
                Please complete your application to access brand resources.
                <a href="/signup/" class="pmh-link pmh-link--no-text-decoration" data-analytics-on-click="{
                                                                                                                    'eventCategory': 'global',
                                                                                                                    'eventAction': 'banner click',
                                                                                                                    'eventLabel': 'complete your application'
                                                                                                                    }"
                   data-ga4-analytics-event="{
                                             'event': 'banner_click',
                                             'link_text': 'Complete your application',
                                             'link_url': '/signup/'
                                             }">
                    Complete your application.
                </a>
            </p>
        </div>
    <!-- Mobile header placeholder -->
        <div class="glue-header__bar glue-header__bar--mobile pmh-header__bar">
            <div class="glue-header__tier">
            <!-- DESKTOP LOCK UP -->
                <div class="glue-header__container">
                    <div class="glue-header__lock-up">
                        <div class="glue-header__hamburger glue-header__hamburger--first-tier">
                            <div class="glue-header__hamburger-wrapper">
                                <button type="button" class="glue-header__drawer-toggle-btn" aria-controls="glue-drawer"
                                        aria-expanded="false" title="Open the navigation drawer"
                                        data-analytics-on-click="{
                                                                 'eventCategory': 'global',
                                                                 'eventAction': 'hamburger menu state click',
                                                                 'eventLabel': 'opened'
                                                                 }"
                                        data-ga4-analytics-event="{
                                                                  'event': 'nav_click',
                                                                  'module_name': 'mobile nav',
                                                                  'link_text': 'Open the navigation drawer',
                                                                  'link_url': '/'
                                                                  }">
                                    <svg aria-hidden="true" class="glue-icon glue-icon--24px">
                                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="m3 18h18v-2h-18zm0-5h18v-2h-18zm0-7v2h18v-2z"/></svg>
                                    </svg>
                                </button>
                            </div>
                        </div>
                    <!-- MOBILE LOCK UP -->
                        <div class="glue-header__lock-up">
                            <div class="glue-header__logo">
                                <a class="glue-header__logo-link" href="https://www.google.com" title="Google"
                                   aria-label="Google link with logo color" target="_blank"
                                   rel="noopener noreferrer" data-analytics-on-click="{
                                                                                      'eventCategory': 'global',
                                                                                      'eventAction': 'nav logo click',
                                                                                      'eventLabel': 'logo'
                                                                                      }"
                                   data-ga4-analytics-event="{
                                                             'event': 'nav_click',
                                                             'module_name': 'main nav',
                                                             'link_text': 'Google link with logo color',
                                                             'link_url': 'https://www.google.com'
                                                             }">
                                    <div class="glue-header__logo-container">
                                        <svg aria-hidden="true" class="glue-header__logo-svg">
                                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 74 24"><path fill="#4285F4" d="M9.24 8.19v2.46h5.88c-.18 1.38-.64 2.39-1.34 3.1-.86.86-2.2 1.8-4.54 1.8-3.62 0-6.45-2.92-6.45-6.54s2.83-6.54 6.45-6.54c1.95 0 3.38.77 4.43 1.76L15.4 2.5C13.94 1.08 11.98 0 9.24 0 4.28 0 .11 4.04.11 9s4.17 9 9.13 9c2.68 0 4.7-.88 6.28-2.52 1.62-1.62 2.13-3.91 2.13-5.75 0-.57-.04-1.1-.13-1.54H9.24z"/><path fill="#EA4335" d="M25 6.19c-3.21 0-5.83 2.44-5.83 5.81 0 3.34 2.62 5.81 5.83 5.81s5.83-2.46 5.83-5.81c0-3.37-2.62-5.81-5.83-5.81zm0 9.33c-1.76 0-3.28-1.45-3.28-3.52 0-2.09 1.52-3.52 3.28-3.52s3.28 1.43 3.28 3.52c0 2.07-1.52 3.52-3.28 3.52z"/><path fill="#4285F4" d="M53.58 7.49h-.09c-.57-.68-1.67-1.3-3.06-1.3C47.53 6.19 45 8.72 45 12c0 3.26 2.53 5.81 5.43 5.81 1.39 0 2.49-.62 3.06-1.32h.09v.81c0 2.22-1.19 3.41-3.1 3.41-1.56 0-2.53-1.12-2.93-2.07l-2.22.92c.64 1.54 2.33 3.43 5.15 3.43 2.99 0 5.52-1.76 5.52-6.05V6.49h-2.42v1zm-2.93 8.03c-1.76 0-3.1-1.5-3.1-3.52 0-2.05 1.34-3.52 3.1-3.52 1.74 0 3.1 1.5 3.1 3.54.01 2.03-1.36 3.5-3.1 3.5z"/><path fill="#FBBC05" d="M38 6.19c-3.21 0-5.83 2.44-5.83 5.81 0 3.34 2.62 5.81 5.83 5.81s5.83-2.46 5.83-5.81c0-3.37-2.62-5.81-5.83-5.81zm0 9.33c-1.76 0-3.28-1.45-3.28-3.52 0-2.09 1.52-3.52 3.28-3.52s3.28 1.43 3.28 3.52c0 2.07-1.52 3.52-3.28 3.52z"/><path fill="#34A853" d="M58 .24h2.51v17.57H58z"/><path fill="#EA4335" d="M68.26 15.52c-1.3 0-2.22-.59-2.82-1.76l7.77-3.21-.26-.66c-.48-1.3-1.96-3.7-4.97-3.7-2.99 0-5.48 2.35-5.48 5.81 0 3.26 2.46 5.81 5.76 5.81 2.66 0 4.2-1.63 4.84-2.57l-1.98-1.32c-.66.96-1.56 1.6-2.86 1.6zm-.18-7.15c1.03 0 1.91.53 2.2 1.28l-5.25 2.17c0-2.44 1.73-3.45 3.05-3.45z"/></svg>
                                        </svg>
                                    </div>
                                </a>
                            </div>
                        </div>
                        <div class="glue-header__logo">
                            <a class="glue-header__logo-link" href="/" title="Partner marketing hub"
                               aria-label="Partner marketing hub" data-analytics-on-click="{
                                                                                                         'eventCategory': 'global',
                                                                                                         'eventAction': 'nav home click',
                                                                                                         'eventLabel': 'home'
                                                                                                         }"
                               data-ga4-analytics-event="{
                                                         'event': 'nav_click',
                                                         'module_name': 'home logo',
                                                         'link_text': 'Partner marketing hub',
                                                         'link_url': '/'
                                                         }">
                                <div class="glue-header__logo-container pmh-header__logo-container">
                                    <svg aria-hidden="true" class="glue-header__logo-svg">
                                        <svg viewBox="0 0 215 22" xmlns="http://www.w3.org/2000/svg"><path d="m2.608 9.62v6.38h-2.024v-15.752h5.368c1.364 0 2.523.455 3.476 1.364.968.91 1.452 2.017 1.452 3.322 0 1.335-.484 2.45-1.452 3.344-.939.895-2.097 1.342-3.476 1.342zm0-7.436v5.5h3.388c.807 0 1.474-.271 2.002-.814.543-.543.814-1.188.814-1.936 0-.733-.271-1.371-.814-1.914-.528-.557-1.195-.836-2.002-.836zm11.066 10.516c0 .528.224.968.671 1.32a2.47 2.47 0 0 0 1.573.528c.85 0 1.61-.315 2.277-.946.667-.63 1.001-1.371 1.001-2.222-.63-.499-1.51-.748-2.64-.748-.821 0-1.507.198-2.057.594s-.825.887-.825 1.474zm2.618-7.832c1.496 0 2.677.4 3.542 1.199.865.8 1.298 1.896 1.298 3.289v6.644h-1.936v-1.496h-.088c-.836 1.232-1.95 1.848-3.344 1.848-1.188 0-2.182-.352-2.981-1.056-.8-.704-1.199-1.584-1.199-2.64 0-1.115.422-2.002 1.265-2.662s1.969-.99 3.377-.99c1.203 0 2.193.22 2.97.66v-.462c0-.704-.279-1.302-.836-1.793a2.867 2.867 0 0 0 -1.958-.737c-1.13 0-2.024.477-2.684 1.43l-1.782-1.122c.983-1.408 2.435-2.112 4.356-2.112zm19.756 11.308c-.88 0-1.61-.271-2.189-.814-.58-.543-.876-1.298-.891-2.266v-6.028h-1.892v-1.848h1.892v-3.3h2.024v3.3h2.64v1.848h-2.64v5.368c0 .719.14 1.206.418 1.463.279.257.594.385.946.385.161 0 .319-.018.473-.055s.297-.084.429-.143l.638 1.804c-.528.19-1.144.286-1.848.286zm-12.628-10.956h1.936v1.76h.088c.601-1.379 1.84-2.068 3.718-2.068l.418.044v2.112l-.902-.088c-.939 0-1.712.29-2.321.869-.609.58-.913 1.316-.913 2.211v5.94h-2.024zm16.258 0h1.936v1.496h.088c.308-.528.781-.968 1.419-1.32a4.067 4.067 0 0 1 1.991-.528c1.32 0 2.336.378 3.047 1.133s1.067 1.83 1.067 3.223v6.776h-2.024v-6.644c-.044-1.76-.931-2.64-2.662-2.64-.807 0-1.481.326-2.024.979s-.814 1.434-.814 2.343v5.962h-2.024zm16.764 11.132c-1.584 0-2.89-.543-3.916-1.628-1.027-1.085-1.54-2.457-1.54-4.114 0-1.643.499-3.01 1.496-4.103s2.273-1.639 3.828-1.639c1.599 0 2.871.517 3.817 1.551s1.419 2.482 1.419 4.345l-.022.22h-8.47c.03 1.056.381 1.907 1.056 2.552a3.379 3.379 0 0 0 2.42.968c1.29 0 2.303-.645 3.036-1.936l1.804.88a5.377 5.377 0 0 1 -2.013 2.134c-.858.513-1.83.77-2.915.77zm-3.234-7.04h6.182a2.684 2.684 0 0 0 -.913-1.859c-.55-.491-1.287-.737-2.211-.737-.763 0-1.419.235-1.969.704-.55.47-.913 1.1-1.089 1.892zm12.21 6.688h-2.024v-10.78h1.936v1.76h.088c.205-.572.627-1.06 1.265-1.463s1.265-.605 1.881-.605c.587 0 1.085.088 1.496.264l-.616 1.958c-.25-.103-.645-.154-1.188-.154-.763 0-1.426.308-1.991.924a3.09 3.09 0 0 0 -.847 2.156zm13.178 0h-2.024v-15.752h2.024l5.478 9.592h.088l5.478-9.592h2.024v15.752h-2.024v-9.35l.088-2.64h-.088l-4.928 8.646h-1.188l-4.928-8.646h-.088l.088 2.64zm17.028-3.3c0 .528.224.968.671 1.32a2.47 2.47 0 0 0 1.573.528c.85 0 1.61-.315 2.277-.946.667-.63 1.001-1.371 1.001-2.222-.63-.499-1.51-.748-2.64-.748-.821 0-1.507.198-2.057.594s-.825.887-.825 1.474zm2.618-7.832c1.496 0 2.677.4 3.542 1.199.865.8 1.298 1.896 1.298 3.289v6.644h-1.936v-1.496h-.088c-.836 1.232-1.95 1.848-3.344 1.848-1.188 0-2.182-.352-2.981-1.056-.8-.704-1.199-1.584-1.199-2.64 0-1.115.422-2.002 1.265-2.662s1.969-.99 3.377-.99c1.203 0 2.193.22 2.97.66v-.462c0-.704-.279-1.302-.836-1.793a2.867 2.867 0 0 0 -1.958-.737c-1.13 0-2.024.477-2.684 1.43l-1.782-1.122c.983-1.408 2.435-2.112 4.356-2.112zm9.152 11.132h-2.024v-10.78h1.936v1.76h.088c.205-.572.627-1.06 1.265-1.463s1.265-.605 1.881-.605c.587 0 1.085.088 1.496.264l-.616 1.958c-.25-.103-.645-.154-1.188-.154-.763 0-1.426.308-1.991.924a3.09 3.09 0 0 0 -.847 2.156zm15.554 0h-2.442l-3.366-5.082-1.65 1.628v3.454h-2.024v-15.752h2.024v9.702l4.664-4.73h2.596v.088l-4.202 4.18 4.4 6.424zm5.654.352c-1.584 0-2.89-.543-3.916-1.628-1.027-1.085-1.54-2.457-1.54-4.114 0-1.643.499-3.01 1.496-4.103s2.273-1.639 3.828-1.639c1.599 0 2.871.517 3.817 1.551s1.419 2.482 1.419 4.345l-.022.22h-8.47c.03 1.056.381 1.907 1.056 2.552a3.379 3.379 0 0 0 2.42.968c1.29 0 2.303-.645 3.036-1.936l1.804.88a5.377 5.377 0 0 1 -2.013 2.134c-.858.513-1.83.77-2.915.77zm-3.234-7.04h6.182a2.684 2.684 0 0 0 -.913-1.859c-.55-.491-1.287-.737-2.211-.737-.763 0-1.419.235-1.969.704-.55.47-.913 1.1-1.089 1.892zm14.124 6.864c-.88 0-1.61-.271-2.189-.814-.58-.543-.876-1.298-.891-2.266v-6.028h-1.892v-1.848h1.892v-3.3h2.024v3.3h2.64v1.848h-2.64v5.368c0 .719.14 1.206.418 1.463.279.257.594.385.946.385.161 0 .319-.018.473-.055s.297-.084.429-.143l.638 1.804c-.528.19-1.144.286-1.848.286zm6.248-14.674c0 .396-.14.733-.418 1.012a1.378 1.378 0 0 1 -1.012.418c-.396 0-.733-.14-1.012-.418a1.378 1.378 0 0 1 -.418-1.012c0-.396.14-.733.418-1.012a1.378 1.378 0 0 1 1.012-.418c.396 0 .733.14 1.012.418.279.279.418.616.418 1.012zm-.418 3.718v10.78h-2.024v-10.78zm2.64 0h1.936v1.496h.088c.308-.528.781-.968 1.419-1.32a4.067 4.067 0 0 1 1.991-.528c1.32 0 2.336.378 3.047 1.133s1.067 1.83 1.067 3.223v6.776h-2.024v-6.644c-.044-1.76-.931-2.64-2.662-2.64-.807 0-1.481.326-2.024.979s-.814 1.434-.814 2.343v5.962h-2.024zm16.654 9.284c.997 0 1.811-.36 2.442-1.078.66-.719.99-1.657.99-2.816 0-1.13-.33-2.06-.99-2.794-.645-.733-1.46-1.1-2.442-1.1-.968 0-1.782.367-2.442 1.1s-.99 1.665-.99 2.794c0 1.144.33 2.075.99 2.794.66.733 1.474 1.1 2.442 1.1zm-.066 6.6a6.023 6.023 0 0 1 -1.694-.231 5.475 5.475 0 0 1 -1.441-.649 4.654 4.654 0 0 1 -1.111-.99 4.174 4.174 0 0 1 -.704-1.276l1.914-.792a3.03 3.03 0 0 0 1.144 1.518c.543.381 1.173.572 1.892.572 1.1 0 1.958-.33 2.574-.99s.924-1.57.924-2.728v-1.034h-.088a3.981 3.981 0 0 1 -1.551 1.353c-.653.33-1.36.495-2.123.495-1.408 0-2.618-.55-3.63-1.65-.997-1.13-1.496-2.493-1.496-4.092s.499-2.955 1.496-4.07c1.012-1.115 2.222-1.672 3.63-1.672.763 0 1.47.165 2.123.495s1.17.781 1.551 1.353h.088v-1.496h1.936v10.318c0 1.73-.491 3.087-1.474 4.07-.997.997-2.317 1.496-3.96 1.496zm15.114-5.104h-2.024v-15.752h2.024v6.754h7.876v-6.754h2.024v15.752h-2.024v-7.106h-7.876zm21.956 0h-1.936v-1.496h-.088c-.308.528-.781.968-1.419 1.32a4.067 4.067 0 0 1 -1.991.528c-1.32 0-2.336-.378-3.047-1.133s-1.067-1.83-1.067-3.223v-6.776h2.024v6.644c.044 1.76.931 2.64 2.662 2.64.807 0 1.481-.326 2.024-.979s.814-1.434.814-2.343v-5.962h2.024zm8.162.352a4.649 4.649 0 0 1 -2.167-.506c-.653-.337-1.155-.785-1.507-1.342h-.088v1.496h-1.936v-15.752h2.024v4.972l-.088 1.496h.088c.352-.557.854-1.005 1.507-1.342a4.649 4.649 0 0 1 2.167-.506c1.423 0 2.625.557 3.608 1.672 1.012 1.13 1.518 2.486 1.518 4.07 0 1.599-.506 2.955-1.518 4.07-.983 1.115-2.185 1.672-3.608 1.672zm-.33-1.848c.968 0 1.782-.367 2.442-1.1.66-.719.99-1.65.99-2.794 0-1.13-.33-2.06-.99-2.794-.66-.733-1.474-1.1-2.442-1.1-.983 0-1.804.367-2.464 1.1-.645.733-.968 1.665-.968 2.794 0 1.144.323 2.083.968 2.816.66.719 1.481 1.078 2.464 1.078z" fill="#5f6368" fill-rule="evenodd"/></svg>
                                    </svg>
                                </div>
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div id="glue-drawer" class="pmh-header__bar glue-header__bar glue-header__bar--desktop glue-header__drawer">
            <div class="glue-header__tier">
                <div class="glue-header__container">
                <!-- LOCK UP-->
                    <div class="glue-header__lock-up">
                    <!-- SKIP -->
                        <a href="#page-content" class="glue-header__link glue-header__skip-content">
                            Jump to Content
                        </a>
                        <div class="glue-header__logo">
                            <a class="glue-header__logo-link" href="https://www.google.com" title="Google"
                               aria-label="Google link with logo color" target="_blank"
                               rel="noopener noreferrer" data-analytics-on-click="{
                                                                                  'eventCategory': 'global',
                                                                                  'eventAction': 'nav logo click',
                                                                                  'eventLabel': 'logo'
                                                                                  }"
                               data-ga4-analytics-event="{
                                                         'event': 'nav_click',
                                                         'module_name': 'main nav',
                                                         'link_text': 'Google link with logo color',
                                                         'link_url': 'https://www.google.com'
                                                         }">
                                <div class="glue-header__logo-container">
                                    <svg aria-hidden="true" class="glue-header__logo-svg">
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 74 24"><path fill="#4285F4" d="M9.24 8.19v2.46h5.88c-.18 1.38-.64 2.39-1.34 3.1-.86.86-2.2 1.8-4.54 1.8-3.62 0-6.45-2.92-6.45-6.54s2.83-6.54 6.45-6.54c1.95 0 3.38.77 4.43 1.76L15.4 2.5C13.94 1.08 11.98 0 9.24 0 4.28 0 .11 4.04.11 9s4.17 9 9.13 9c2.68 0 4.7-.88 6.28-2.52 1.62-1.62 2.13-3.91 2.13-5.75 0-.57-.04-1.1-.13-1.54H9.24z"/><path fill="#EA4335" d="M25 6.19c-3.21 0-5.83 2.44-5.83 5.81 0 3.34 2.62 5.81 5.83 5.81s5.83-2.46 5.83-5.81c0-3.37-2.62-5.81-5.83-5.81zm0 9.33c-1.76 0-3.28-1.45-3.28-3.52 0-2.09 1.52-3.52 3.28-3.52s3.28 1.43 3.28 3.52c0 2.07-1.52 3.52-3.28 3.52z"/><path fill="#4285F4" d="M53.58 7.49h-.09c-.57-.68-1.67-1.3-3.06-1.3C47.53 6.19 45 8.72 45 12c0 3.26 2.53 5.81 5.43 5.81 1.39 0 2.49-.62 3.06-1.32h.09v.81c0 2.22-1.19 3.41-3.1 3.41-1.56 0-2.53-1.12-2.93-2.07l-2.22.92c.64 1.54 2.33 3.43 5.15 3.43 2.99 0 5.52-1.76 5.52-6.05V6.49h-2.42v1zm-2.93 8.03c-1.76 0-3.1-1.5-3.1-3.52 0-2.05 1.34-3.52 3.1-3.52 1.74 0 3.1 1.5 3.1 3.54.01 2.03-1.36 3.5-3.1 3.5z"/><path fill="#FBBC05" d="M38 6.19c-3.21 0-5.83 2.44-5.83 5.81 0 3.34 2.62 5.81 5.83 5.81s5.83-2.46 5.83-5.81c0-3.37-2.62-5.81-5.83-5.81zm0 9.33c-1.76 0-3.28-1.45-3.28-3.52 0-2.09 1.52-3.52 3.28-3.52s3.28 1.43 3.28 3.52c0 2.07-1.52 3.52-3.28 3.52z"/><path fill="#34A853" d="M58 .24h2.51v17.57H58z"/><path fill="#EA4335" d="M68.26 15.52c-1.3 0-2.22-.59-2.82-1.76l7.77-3.21-.26-.66c-.48-1.3-1.96-3.7-4.97-3.7-2.99 0-5.48 2.35-5.48 5.81 0 3.26 2.46 5.81 5.76 5.81 2.66 0 4.2-1.63 4.84-2.57l-1.98-1.32c-.66.96-1.56 1.6-2.86 1.6zm-.18-7.15c1.03 0 1.91.53 2.2 1.28l-5.25 2.17c0-2.44 1.73-3.45 3.05-3.45z"/></svg>
                                    </svg>
                                </div>
                            </a>
                        </div>
                        <div class="glue-header__logo">
                            <a class="glue-header__logo-link" href="/" title="Partner marketing hub"
                               aria-label="Partner marketing hub logo" data-analytics-on-click="{
                                                                                                              'eventCategory': 'global',
                                                                                                              'eventAction': 'nav home click',
                                                                                                              'eventLabel': 'home'
                                                                                                              }"
                               data-ga4-analytics-event="{
                                                         'event': 'nav_click',
                                                         'module_name': 'home logo',
                                                         'link_text': 'Partner marketing hub logo',
                                                         'link_url': '/'
                                                         }">
                                <div class="glue-header__logo-container pmh-header__logo-container">
                                    <svg aria-hidden="true" class="glue-header__logo-svg">
                                        <svg viewBox="0 0 215 22" xmlns="http://www.w3.org/2000/svg"><path d="m2.608 9.62v6.38h-2.024v-15.752h5.368c1.364 0 2.523.455 3.476 1.364.968.91 1.452 2.017 1.452 3.322 0 1.335-.484 2.45-1.452 3.344-.939.895-2.097 1.342-3.476 1.342zm0-7.436v5.5h3.388c.807 0 1.474-.271 2.002-.814.543-.543.814-1.188.814-1.936 0-.733-.271-1.371-.814-1.914-.528-.557-1.195-.836-2.002-.836zm11.066 10.516c0 .528.224.968.671 1.32a2.47 2.47 0 0 0 1.573.528c.85 0 1.61-.315 2.277-.946.667-.63 1.001-1.371 1.001-2.222-.63-.499-1.51-.748-2.64-.748-.821 0-1.507.198-2.057.594s-.825.887-.825 1.474zm2.618-7.832c1.496 0 2.677.4 3.542 1.199.865.8 1.298 1.896 1.298 3.289v6.644h-1.936v-1.496h-.088c-.836 1.232-1.95 1.848-3.344 1.848-1.188 0-2.182-.352-2.981-1.056-.8-.704-1.199-1.584-1.199-2.64 0-1.115.422-2.002 1.265-2.662s1.969-.99 3.377-.99c1.203 0 2.193.22 2.97.66v-.462c0-.704-.279-1.302-.836-1.793a2.867 2.867 0 0 0 -1.958-.737c-1.13 0-2.024.477-2.684 1.43l-1.782-1.122c.983-1.408 2.435-2.112 4.356-2.112zm19.756 11.308c-.88 0-1.61-.271-2.189-.814-.58-.543-.876-1.298-.891-2.266v-6.028h-1.892v-1.848h1.892v-3.3h2.024v3.3h2.64v1.848h-2.64v5.368c0 .719.14 1.206.418 1.463.279.257.594.385.946.385.161 0 .319-.018.473-.055s.297-.084.429-.143l.638 1.804c-.528.19-1.144.286-1.848.286zm-12.628-10.956h1.936v1.76h.088c.601-1.379 1.84-2.068 3.718-2.068l.418.044v2.112l-.902-.088c-.939 0-1.712.29-2.321.869-.609.58-.913 1.316-.913 2.211v5.94h-2.024zm16.258 0h1.936v1.496h.088c.308-.528.781-.968 1.419-1.32a4.067 4.067 0 0 1 1.991-.528c1.32 0 2.336.378 3.047 1.133s1.067 1.83 1.067 3.223v6.776h-2.024v-6.644c-.044-1.76-.931-2.64-2.662-2.64-.807 0-1.481.326-2.024.979s-.814 1.434-.814 2.343v5.962h-2.024zm16.764 11.132c-1.584 0-2.89-.543-3.916-1.628-1.027-1.085-1.54-2.457-1.54-4.114 0-1.643.499-3.01 1.496-4.103s2.273-1.639 3.828-1.639c1.599 0 2.871.517 3.817 1.551s1.419 2.482 1.419 4.345l-.022.22h-8.47c.03 1.056.381 1.907 1.056 2.552a3.379 3.379 0 0 0 2.42.968c1.29 0 2.303-.645 3.036-1.936l1.804.88a5.377 5.377 0 0 1 -2.013 2.134c-.858.513-1.83.77-2.915.77zm-3.234-7.04h6.182a2.684 2.684 0 0 0 -.913-1.859c-.55-.491-1.287-.737-2.211-.737-.763 0-1.419.235-1.969.704-.55.47-.913 1.1-1.089 1.892zm12.21 6.688h-2.024v-10.78h1.936v1.76h.088c.205-.572.627-1.06 1.265-1.463s1.265-.605 1.881-.605c.587 0 1.085.088 1.496.264l-.616 1.958c-.25-.103-.645-.154-1.188-.154-.763 0-1.426.308-1.991.924a3.09 3.09 0 0 0 -.847 2.156zm13.178 0h-2.024v-15.752h2.024l5.478 9.592h.088l5.478-9.592h2.024v15.752h-2.024v-9.35l.088-2.64h-.088l-4.928 8.646h-1.188l-4.928-8.646h-.088l.088 2.64zm17.028-3.3c0 .528.224.968.671 1.32a2.47 2.47 0 0 0 1.573.528c.85 0 1.61-.315 2.277-.946.667-.63 1.001-1.371 1.001-2.222-.63-.499-1.51-.748-2.64-.748-.821 0-1.507.198-2.057.594s-.825.887-.825 1.474zm2.618-7.832c1.496 0 2.677.4 3.542 1.199.865.8 1.298 1.896 1.298 3.289v6.644h-1.936v-1.496h-.088c-.836 1.232-1.95 1.848-3.344 1.848-1.188 0-2.182-.352-2.981-1.056-.8-.704-1.199-1.584-1.199-2.64 0-1.115.422-2.002 1.265-2.662s1.969-.99 3.377-.99c1.203 0 2.193.22 2.97.66v-.462c0-.704-.279-1.302-.836-1.793a2.867 2.867 0 0 0 -1.958-.737c-1.13 0-2.024.477-2.684 1.43l-1.782-1.122c.983-1.408 2.435-2.112 4.356-2.112zm9.152 11.132h-2.024v-10.78h1.936v1.76h.088c.205-.572.627-1.06 1.265-1.463s1.265-.605 1.881-.605c.587 0 1.085.088 1.496.264l-.616 1.958c-.25-.103-.645-.154-1.188-.154-.763 0-1.426.308-1.991.924a3.09 3.09 0 0 0 -.847 2.156zm15.554 0h-2.442l-3.366-5.082-1.65 1.628v3.454h-2.024v-15.752h2.024v9.702l4.664-4.73h2.596v.088l-4.202 4.18 4.4 6.424zm5.654.352c-1.584 0-2.89-.543-3.916-1.628-1.027-1.085-1.54-2.457-1.54-4.114 0-1.643.499-3.01 1.496-4.103s2.273-1.639 3.828-1.639c1.599 0 2.871.517 3.817 1.551s1.419 2.482 1.419 4.345l-.022.22h-8.47c.03 1.056.381 1.907 1.056 2.552a3.379 3.379 0 0 0 2.42.968c1.29 0 2.303-.645 3.036-1.936l1.804.88a5.377 5.377 0 0 1 -2.013 2.134c-.858.513-1.83.77-2.915.77zm-3.234-7.04h6.182a2.684 2.684 0 0 0 -.913-1.859c-.55-.491-1.287-.737-2.211-.737-.763 0-1.419.235-1.969.704-.55.47-.913 1.1-1.089 1.892zm14.124 6.864c-.88 0-1.61-.271-2.189-.814-.58-.543-.876-1.298-.891-2.266v-6.028h-1.892v-1.848h1.892v-3.3h2.024v3.3h2.64v1.848h-2.64v5.368c0 .719.14 1.206.418 1.463.279.257.594.385.946.385.161 0 .319-.018.473-.055s.297-.084.429-.143l.638 1.804c-.528.19-1.144.286-1.848.286zm6.248-14.674c0 .396-.14.733-.418 1.012a1.378 1.378 0 0 1 -1.012.418c-.396 0-.733-.14-1.012-.418a1.378 1.378 0 0 1 -.418-1.012c0-.396.14-.733.418-1.012a1.378 1.378 0 0 1 1.012-.418c.396 0 .733.14 1.012.418.279.279.418.616.418 1.012zm-.418 3.718v10.78h-2.024v-10.78zm2.64 0h1.936v1.496h.088c.308-.528.781-.968 1.419-1.32a4.067 4.067 0 0 1 1.991-.528c1.32 0 2.336.378 3.047 1.133s1.067 1.83 1.067 3.223v6.776h-2.024v-6.644c-.044-1.76-.931-2.64-2.662-2.64-.807 0-1.481.326-2.024.979s-.814 1.434-.814 2.343v5.962h-2.024zm16.654 9.284c.997 0 1.811-.36 2.442-1.078.66-.719.99-1.657.99-2.816 0-1.13-.33-2.06-.99-2.794-.645-.733-1.46-1.1-2.442-1.1-.968 0-1.782.367-2.442 1.1s-.99 1.665-.99 2.794c0 1.144.33 2.075.99 2.794.66.733 1.474 1.1 2.442 1.1zm-.066 6.6a6.023 6.023 0 0 1 -1.694-.231 5.475 5.475 0 0 1 -1.441-.649 4.654 4.654 0 0 1 -1.111-.99 4.174 4.174 0 0 1 -.704-1.276l1.914-.792a3.03 3.03 0 0 0 1.144 1.518c.543.381 1.173.572 1.892.572 1.1 0 1.958-.33 2.574-.99s.924-1.57.924-2.728v-1.034h-.088a3.981 3.981 0 0 1 -1.551 1.353c-.653.33-1.36.495-2.123.495-1.408 0-2.618-.55-3.63-1.65-.997-1.13-1.496-2.493-1.496-4.092s.499-2.955 1.496-4.07c1.012-1.115 2.222-1.672 3.63-1.672.763 0 1.47.165 2.123.495s1.17.781 1.551 1.353h.088v-1.496h1.936v10.318c0 1.73-.491 3.087-1.474 4.07-.997.997-2.317 1.496-3.96 1.496zm15.114-5.104h-2.024v-15.752h2.024v6.754h7.876v-6.754h2.024v15.752h-2.024v-7.106h-7.876zm21.956 0h-1.936v-1.496h-.088c-.308.528-.781.968-1.419 1.32a4.067 4.067 0 0 1 -1.991.528c-1.32 0-2.336-.378-3.047-1.133s-1.067-1.83-1.067-3.223v-6.776h2.024v6.644c.044 1.76.931 2.64 2.662 2.64.807 0 1.481-.326 2.024-.979s.814-1.434.814-2.343v-5.962h2.024zm8.162.352a4.649 4.649 0 0 1 -2.167-.506c-.653-.337-1.155-.785-1.507-1.342h-.088v1.496h-1.936v-15.752h2.024v4.972l-.088 1.496h.088c.352-.557.854-1.005 1.507-1.342a4.649 4.649 0 0 1 2.167-.506c1.423 0 2.625.557 3.608 1.672 1.012 1.13 1.518 2.486 1.518 4.07 0 1.599-.506 2.955-1.518 4.07-.983 1.115-2.185 1.672-3.608 1.672zm-.33-1.848c.968 0 1.782-.367 2.442-1.1.66-.719.99-1.65.99-2.794 0-1.13-.33-2.06-.99-2.794-.66-.733-1.474-1.1-2.442-1.1-.983 0-1.804.367-2.464 1.1-.645.733-.968 1.665-.968 2.794 0 1.144.323 2.083.968 2.816.66.719 1.481 1.078 2.464 1.078z" fill="#5f6368" fill-rule="evenodd"/></svg>
                                    </svg>
                                </div>
                            </a>
                        </div>
                    </div>
                </div>
            <!-- LINK BAR-->
                <div class="glue-header__container glue-header__container--linkbar">
                    
                    <nav class="glue-header__link-bar">
                        <ul class="glue-header__list">
                            <li
                                class="glue-header__item  glue-header__item--active ">
                                <a class="glue-header__link pmh-header__link" href="/brands/" id="subnav-brand__button"
                                   aria-controls="subnav-brand" aria-expanded="false" data-target="#subnav-brand"
                                   data-is-link aria-haspopup="true" data-analytics-on-click="{
                                                                                                                              'eventCategory': 'global',
                                                                                                                              'eventAction': 'nav link click',
                                                                                                                              'eventLabel': 'brands'
                                                                                                                              }" data-ga4-analytics-event="{
                                                                                                                            'event': 'nav_click',
                                                                                                                            'module_name': 'main nav',
                                                                                                                            'link_text': 'Brands',
                                                                                                                            'link_url': '/brands/'
                                                                                                                            }">
                                    Brands

                                    <svg aria-hidden="true" class="glue-icon pmh-header__nav-icon pmh-hide--mobile">
                                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M5.41 7.59L4 9l8 8 8-8-1.41-1.41L12 14.17"></path></svg>

                                    </svg>
                                </a>
                            </li>
                            <li
                                class="glue-header__item ">
                                <a class="glue-header__link pmh-header__link" href="/marketing-opportunities/"
                                   data-analytics-on-click="{
                                                            'eventCategory': 'global',
                                                            'eventAction': 'nav link click',
                                                            'eventLabel': 'marketing opportunities'
                                                            }" data-ga4-analytics-event="{
                                                                                                                            'event': 'nav_click',
                                                                                                                            'module_name': 'main nav',
                                                                                                                            'link_text': 'Marketing opportunities',
                                                                                                                            'link_url': '/marketing-opportunities/'
                                                                                                                            }">
                                    Marketing opportunities
                                </a>
                            </li>
                            <li
                                class="glue-header__item ">
                                <a class="glue-header__link pmh-header__link" href="/tools/"
                                   data-analytics-on-click="{
                                                            'eventCategory': 'global',
                                                            'eventAction': 'nav link click',
                                                            'eventLabel': 'tools'
                                                            }"
                                   data-ga4-analytics-event="{
                                                             'event': 'nav_click',
                                                             'module_name': 'main nav',
                                                             'link_text': 'Tools',
                                                             'link_url': '/tools/'
                                                             }">
                                    Tools
                                </a>
                            </li>
                            <li
                                class="glue-header__item ">
                                <a class="glue-header__link pmh-header__link" href="/asset-approval/"
                                   data-analytics-on-click="{
                                                            'eventCategory': 'global',
                                                            'eventAction': 'nav link click',
                                                            'eventLabel': 'asset approval'
                                                            }" data-ga4-analytics-event="{
                                                                                                                            'event': 'nav_click',
                                                                                                                            'module_name': 'main nav',
                                                                                                                            'link_text': 'Asset approval',
                                                                                                                            'link_url': '/asset-approval/'
                                                                                                                            }">
                                    Asset approval
                                </a>
                            </li>
                            <li
                                class="glue-header__item ">
                            
                                <div class="subnav-menu pmh-hide--mobile">
                                    <button class="subnav-menu__trigger glue-header__link pmh-header__link"
                                            aria-controls="subnav-menu-support" aria-expanded="false"
                                            data-target="#subnav-menu-support" aria-haspopup="true" tabindex="0">
                                        <span>Support</span>
                                        <svg aria-hidden="true" class="glue-icon pmh-header__nav-icon pmh-hide--mobile">
                                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M5.41 7.59L4 9l8 8 8-8-1.41-1.41L12 14.17"></path></svg>

                                        </svg>
                                    </button>
                                    <div class="subnav-menu__wrapper" id="subnav-menu-support" role="dialog"
                                         aria-hidden="true" inert>
                                        <div class="subnav-menu__content">
                                            <div class="subnav-menu__link-list">
                                                <a class="subnav-menu__link" href="/support/contact-us"
                                                   data-analytics-on-click="{
                                                                            'eventCategory': 'global',
                                                                            'eventAction': 'nav link click',
                                                                            'eventLabel': 'contact_us'
                                                                            }" data-ga4-analytics-event="{
                                                                                                                                                            'event': 'nav_click',
                                                                                                                                                            'module_name': 'main nav',
                                                                                                                                                            'link_text': 'Contact us',
                                                                                                                                                            'link_url': '/support/contact-us/'
                                                                                                                                                            }">
                                                    <span>Contact us</span>
                                                </a>
                                                <a class="subnav-menu__link" href="/support/faq" data-analytics-on-click="{
                                                                                                                          'eventCategory': 'global',
                                                                                                                          'eventAction': 'nav link click',
                                                                                                                          'eventLabel': 'faq'
                                                                                                                          }" data-ga4-analytics-event="{
                                                                                                                                                            'event': 'nav_click',
                                                                                                                                                            'module_name': 'main nav',
                                                                                                                                                            'link_text': 'FAQ',
                                                                                                                                                            'link_url': '/support/faq/'
                                                                                                                                                            }">
                                                    <span>FAQ</span>
                                                </a>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                            
                                <div class="glue-header__item pmh-hide--desktop">
                                    <div class="pmh-header__nested-item">
                                        <span class="glue-header__link pmh-header__nested-item-title">
                                            Support
                                        </span>
                                        <div class="pmh-header__nested-item-links">
                                            <a data-analytics-on-click="{
                                                                        'eventCategory': 'global',
                                                                        'eventAction': 'nav button click',
                                                                        'eventLabel': 'feedback - modal'
                                                                        }" data-ga4-analytics-event="{
                                                                                                                                                    'event': 'nav_click',
                                                                                                                                                    'module_name': 'main nav',
                                                                                                                                                    'link_text': 'Contact us',
                                                                                                                                                    'link_url': '/support/contact-us/'
                                                                                                                                                    }" class="glue-header__link"
                                               href="/support/contact-us/">
                                                <span>Contact us</span>
                                            </a>
                                            <a data-analytics-on-click="{
                                                                        'eventCategory': 'global',
                                                                        'eventAction': 'nav button click',
                                                                        'eventLabel': 'feedback - modal'
                                                                        }" data-ga4-analytics-event="{
                                                                                                                                                    'event': 'nav_click',
                                                                                                                                                    'module_name': 'main nav',
                                                                                                                                                    'link_text': 'FAQ',
                                                                                                                                                    'link_url': '/support/faq/'
                                                                                                                                                    }" class="glue-header__link"
                                               href="/support/faq/">
                                                <span>FAQ</span>
                                            </a>
                                        </div>
                                    </div>
                                </div>
                            </li>
                        </ul>
                    </nav>
                </div>
                <div class="glue-header__container glue-header__container--cta">
                    <div class="glue-header__cta pmh-header__log-in-actions">
                        <div class="glue-header__item">
                            <hds-cta
                                class="pmh-header__login-button hds-button-mte hds-button-mte--outline hds-button-mte--32"
                                href="/signup/" label="Sign up"></hds-cta>
                        </div>
                        <div class="glue-header__item">
                            <hds-cta
                                class="pmh-header__login-button hds-button-mte hds-button-mte--filled hds-button-mte--32"
                                href="/login/" label="Log in"></hds-cta>
                        </div>
                    </div>
                    <div class="glue-header__cta pmh-header__cta pmh-header__logged-actions">
                        <div class="glue-header__item">
                            <a class="pmh-nav__search glue-header__link" href="/search/"
                               title="Search" data-analytics-on-click="{
                                                                                     'eventCategory': 'global',
                                                                                     'eventAction': 'nav icon click',
                                                                                     'eventLabel': 'search'
                                                                                     }"
                               data-ga4-analytics-event="{
                                                         'event': 'nav_click',
                                                         'module_name': 'main nav',
                                                         'link_text': 'Search',
                                                         'link_url': '/search/'
                                                         }">
                                <span class="nav__items__item__text nav__search-text pmh-visually-hidden--desktop">
                                    Search
                                </span>
                                <i class="material-icons" aria-hidden="true">search</i>
                            </a>
                        </div>
                    
                        <div class="subnav-menu pmh-hide--mobile">
                            <button
                                class="subnav-menu__trigger glue-button glue-button--low-emphasis notifications-badge notifications__main-badge"
                                aria-controls="subnav-menu-notifications" aria-expanded="false"
                                data-target="#subnav-menu-notifications" aria-haspopup="true"
                                title="Notifications" tabindex="0">
                                <i class="material-icons" aria-hidden="true">notifications</i>
                            </button>
                            <div class="subnav-menu__wrapper" id="subnav-menu-notifications" role="dialog"
                                 aria-hidden="true" inert>
                                <div class="subnav-menu__content">
                                    <div class="subnav-menu__link-list">
                                        <a class="subnav-menu__link notifications__notifications-badge"
                                           href="/notifications/" data-analytics-on-click="{
                                                                                                               'eventCategory': 'global',
                                                                                                               'eventAction': 'nav button click',
                                                                                                               'eventLabel': 'notifications'
                                                                                                               }"
                                           data-ga4-analytics-event="{
                                                                     'event': 'nav_click',
                                                                     'module_name': 'main nav',
                                                                     'link_text': 'Notifications',
                                                                     'link_url': '/notifications/'
                                                                     }">
                                            <span>Notifications</span>
                                        </a>
                                        
                                        
                                        <a class="subnav-menu__link" href="/submissions/"
                                           data-analytics-on-click="{
                                                                    'eventCategory': 'global',
                                                                    'eventAction': 'nav link click',
                                                                    'eventLabel': 'submissions'
                                                                    }" data-ga4-analytics-event="{
                                                                                                                                            'event': 'nav_click',
                                                                                                                                            'module_name': 'main nav',
                                                                                                                                            'link_text': 'Submissions',
                                                                                                                                            'link_url': '/submissions/'
                                                                                                                                            }">
                                            <span>Submissions</span>
                                        </a>
                                    </div>
                                </div>
                            </div>
                        </div>

                    
                        <div class="glue-header__item pmh-hide--desktop">
                            <div class="pmh-header__nested-item notifications-badge">
                                <span class="glue-header__link pmh-header__nested-item-title">
                                    Notifications
                                    <i class="material-icons" aria-hidden="true">notifications</i>
                                </span>
                                <div class="pmh-header__nested-item-links">
                                    <a class="glue-header__link notifications__notifications-badge"
                                       href="/notifications/" data-analytics-on-click="{
                                                                                                           'eventCategory': 'global',
                                                                                                           'eventAction': 'nav button click',
                                                                                                           'eventLabel': 'notifications'
                                                                                                           }"
                                       data-ga4-analytics-event="{
                                                                 'event': 'nav_click',
                                                                 'module_name': 'main nav',
                                                                 'link_text': 'Notifications',
                                                                 'link_url': '/notifications/'
                                                                 }">
                                        <span>Notifications</span>
                                    </a>
                                    
                                    
                                    <a class="glue-header__link" href="/submissions/"
                                       data-analytics-on-click="{
                                                                'eventCategory': 'global',
                                                                'eventAction': 'nav link click',
                                                                'eventLabel': 'submissions'
                                                                }" data-ga4-analytics-event="{
                                                                                                                                    'event': 'nav_click',
                                                                                                                                    'module_name': 'main nav',
                                                                                                                                    'link_text': 'Submissions',
                                                                                                                                    'link_url': '/submissions/'
                                                                                                                                    }">
                                        <span>Submissions</span>
                                    </a>
                                </div>
                            </div>
                        </div>

                        <div class="glue-header__item">
                            <a class="pmh-button--profile glue-header__link" href="/profile/"
                               title="My account" data-analytics-on-click="{
                                                                                         'eventCategory': 'global',
                                                                                         'eventAction': 'nav icon click',
                                                                                         'eventLabel': 'my account'
                                                                                         }" data-ga4-analytics-event="{
                                                                                                                                                     'event': 'nav_click',
                                                                                                                                                     'module_name': 'main nav',
                                                                                                                                                     'link_text': 'My account',
                                                                                                                                                     'link_url': '/profile/'
                                                                                                                                                     }">
                                <span class="mobile-label pmh-visually-hidden--desktop">My account</span>
                                <span class="pmh-icon__container pmh-rounded">
                                    <svg aria-hidden="true" class="glue-icon pmh-icon--person">
                                        <svg viewBox="0 0 100 122" xmlns="http://www.w3.org/2000/svg">
                                            <path fill="#ffffffAA" fill-rule="nonzero" d="M50 48c13.26 0 24-10.74 24-24S63.26 0 50 0 26 10.74
                                                                                          26 24s10.74 24 24 24zm0 13c16.688 0 50 8.487 50
                                                                                          25.333V122H0V86.333C0 69.487 33.313 61 50 61z" />
                                        </svg>
                                    </svg>
                                </span>
                            </a>
                        </div>
                    </div>
                </div>
            </div>
            
            
            
                <div class="brand_submenu_desktop">
                    
                    
                        <div class="nav__subnav" id="subnav-brand" role="dialog" aria-labelledby="subnav-brand__button"
                             aria-hidden="true">
                            <div class="nav-brands glue-page">
                                <div class="nav-brands__columns"
                                     style="grid-template-columns: repeat(4, 1fr);">
                                    
                                        
                                            <div class="nav-brands__group nav-brands__group--x1">
                                                <h3 class="nav-brands__group__title glue-headline glue-headline--headline-6">
                                                    <a class="glue-header__link glue-font-weight-bold"
                                                       href="/brands#essential_guidelines"
                                                       data-analytics-on-click="{
                                                                                'eventCategory': 'global',
                                                                                'eventAction': 'nav submenu title click',
                                                                                'eventLabel': 'brands'
                                                                                }"
                                                       data-ga4-analytics-event="{
                                                                                 'event': 'nav_click',
                                                                                 'module_name': 'main nav',
                                                                                 'link_text': 'Essential guidelines',
                                                                                 'link_url': '/brands#essential_guidelines'
                                                                                 }">
                                                        Essential guidelines
                                                    </a>
                                                </h3>
                                                <ul class="nav-brands__group__list glue-no-bullet">
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/diversity-inclusion/diversity-equity-inclusion/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Diversity and inclusion'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Diversity and inclusion',
                                                                                             'link_url': '/brands/diversity-inclusion/diversity-equity-inclusion/'
                                                                                             }">
                                                                    Diversity and inclusion
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google',
                                                                                             'link_url': '/brands/google/overview/'
                                                                                             }">
                                                                    Google
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/nd4c-guidance/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'ND4C Guidance'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'ND4C Guidance',
                                                                                             'link_url': '/brands/nd4c-guidance/'
                                                                                             }">
                                                                    ND4C Guidance
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                </ul>
                                            </div>
                                        
                                    
                                        
                                            <div class="nav-brands__group nav-brands__group--x2">
                                                <h3 class="nav-brands__group__title glue-headline glue-headline--headline-6">
                                                    <a class="glue-header__link glue-font-weight-bold"
                                                       href="/brands#platforms"
                                                       data-analytics-on-click="{
                                                                                'eventCategory': 'global',
                                                                                'eventAction': 'nav submenu title click',
                                                                                'eventLabel': 'brands'
                                                                                }"
                                                       data-ga4-analytics-event="{
                                                                                 'event': 'nav_click',
                                                                                 'module_name': 'main nav',
                                                                                 'link_text': 'Platforms',
                                                                                 'link_url': '/brands#platforms'
                                                                                 }">
                                                        Platforms
                                                    </a>
                                                </h3>
                                                <ul class="nav-brands__group__list glue-no-bullet">
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/android/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Android'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Android',
                                                                                             'link_url': '/brands/android/overview/brand-introduction/'
                                                                                             }">
                                                                    Android
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/android-auto/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Android Auto'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Android Auto',
                                                                                             'link_url': '/brands/android-auto/overview/brand-introduction/'
                                                                                             }">
                                                                    Android Auto
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/android-enterprise/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Android Enterprise'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Android Enterprise',
                                                                                             'link_url': '/brands/android-enterprise/overview/brand-introduction/'
                                                                                             }">
                                                                    Android Enterprise
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/android-tv/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Android TV'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Android TV',
                                                                                             'link_url': '/brands/android-tv/overview/brand-introduction/'
                                                                                             }">
                                                                    Android TV
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/android-xr/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Android XR'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Android XR',
                                                                                             'link_url': '/brands/android-xr/overview/'
                                                                                             }">
                                                                    Android XR
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/cars-with-google-built-in/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Cars with Google built\u002Din'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Cars with Google built-in',
                                                                                             'link_url': '/brands/cars-with-google-built-in/overview/brand-introduction/'
                                                                                             }">
                                                                    Cars with Google built-in
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/chrome/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Chrome'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Chrome',
                                                                                             'link_url': '/brands/chrome/overview/brand-introduction/'
                                                                                             }">
                                                                    Chrome
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/chromebook/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Chromebook'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Chromebook',
                                                                                             'link_url': '/brands/chromebook/overview/brand-introduction/'
                                                                                             }">
                                                                    Chromebook
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-tv/overview/google-tv-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google TV'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google TV',
                                                                                             'link_url': '/brands/google-tv/overview/google-tv-introduction/'
                                                                                             }">
                                                                    Google TV
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/wear-os-by-google/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Wear OS by Google'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Wear OS by Google',
                                                                                             'link_url': '/brands/wear-os-by-google/overview/brand-introduction/'
                                                                                             }">
                                                                    Wear OS by Google
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                </ul>
                                            </div>
                                        
                                    
                                        
                                            <div class="nav-brands__group nav-brands__group--x3">
                                                <h3 class="nav-brands__group__title glue-headline glue-headline--headline-6">
                                                    <a class="glue-header__link glue-font-weight-bold"
                                                       href="/brands#apps_and_services"
                                                       data-analytics-on-click="{
                                                                                'eventCategory': 'global',
                                                                                'eventAction': 'nav submenu title click',
                                                                                'eventLabel': 'brands'
                                                                                }"
                                                       data-ga4-analytics-event="{
                                                                                 'event': 'nav_click',
                                                                                 'module_name': 'main nav',
                                                                                 'link_text': 'Apps and services',
                                                                                 'link_url': '/brands#apps_and_services'
                                                                                 }">
                                                        Apps and services
                                                    </a>
                                                </h3>
                                                <ul class="nav-brands__group__list glue-no-bullet">
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/family-link/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Family Link'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Family Link',
                                                                                             'link_url': '/brands/family-link/overview/brand-introduction/'
                                                                                             }">
                                                                    Family Link
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/fast-pair/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Fast Pair'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Fast Pair',
                                                                                             'link_url': '/brands/fast-pair/overview/'
                                                                                             }">
                                                                    Fast Pair
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/find-hub/find-hub/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Find Hub'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Find Hub',
                                                                                             'link_url': '/brands/find-hub/find-hub/'
                                                                                             }">
                                                                    Find Hub
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-assistant/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Assistant'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Assistant',
                                                                                             'link_url': '/brands/google-assistant/overview/brand-introduction/'
                                                                                             }">
                                                                    Google Assistant
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-cast/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Cast'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Cast',
                                                                                             'link_url': '/brands/google-cast/overview/brand-introduction/'
                                                                                             }">
                                                                    Google Cast
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-gemini/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Gemini'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Gemini',
                                                                                             'link_url': '/brands/google-gemini/brand-introduction/'
                                                                                             }">
                                                                    Google Gemini
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-lens/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Lens'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Lens',
                                                                                             'link_url': '/brands/google-lens/overview/brand-introduction/'
                                                                                             }">
                                                                    Google Lens
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-maps/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Maps'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Maps',
                                                                                             'link_url': '/brands/google-maps/overview/brand-introduction/'
                                                                                             }">
                                                                    Google Maps
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-meet/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Meet'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Meet',
                                                                                             'link_url': '/brands/google-meet/overview/brand-introduction/'
                                                                                             }">
                                                                    Google Meet
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-messages/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Messages'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Messages',
                                                                                             'link_url': '/brands/google-messages/overview/'
                                                                                             }">
                                                                    Google Messages
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-news/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google News'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google News',
                                                                                             'link_url': '/brands/google-news/overview/brand-introduction/'
                                                                                             }">
                                                                    Google News
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-one/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google One'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google One',
                                                                                             'link_url': '/brands/google-one/overview/'
                                                                                             }">
                                                                    Google One
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-pay-google-wallet/how-our-brands-work-together/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Pay \u0026 Google Wallet'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Pay &amp; Google Wallet',
                                                                                             'link_url': '/brands/google-pay-google-wallet/how-our-brands-work-together/'
                                                                                             }">
                                                                    Google Pay &amp; Google Wallet
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-photos/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Photos'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Photos',
                                                                                             'link_url': '/brands/google-photos/brand-introduction/'
                                                                                             }">
                                                                    Google Photos
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-play/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Play'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Play',
                                                                                             'link_url': '/brands/google-play/overview/brand-introduction/'
                                                                                             }">
                                                                    Google Play
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-play-protect/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Play Protect'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Play Protect',
                                                                                             'link_url': '/brands/google-play-protect/overview/brand-introduction/'
                                                                                             }">
                                                                    Google Play Protect
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/quick-share/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Quick Share'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Quick Share',
                                                                                             'link_url': '/brands/quick-share/overview/'
                                                                                             }">
                                                                    Quick Share
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/waze/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Waze'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Waze',
                                                                                             'link_url': '/brands/waze/overview/'
                                                                                             }">
                                                                    Waze
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/works-with-google-home/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Works With Google Home'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Works With Google Home',
                                                                                             'link_url': '/brands/works-with-google-home/overview/brand-introduction/'
                                                                                             }">
                                                                    Works With Google Home
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/youtube/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'YouTube'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'YouTube',
                                                                                             'link_url': '/brands/youtube/overview/'
                                                                                             }">
                                                                    YouTube
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/youtube-kids/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'YouTube Kids'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'YouTube Kids',
                                                                                             'link_url': '/brands/youtube-kids/overview/brand-introduction/'
                                                                                             }">
                                                                    YouTube Kids
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/youtube-music/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'YouTube Music'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'YouTube Music',
                                                                                             'link_url': '/brands/youtube-music/overview/'
                                                                                             }">
                                                                    YouTube Music
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/youtube-premium/introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'YouTube Premium'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'YouTube Premium',
                                                                                             'link_url': '/brands/youtube-premium/introduction/'
                                                                                             }">
                                                                    YouTube Premium
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/youtube-tv/overview/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'YouTube TV'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'YouTube TV',
                                                                                             'link_url': '/brands/youtube-tv/overview/brand-introduction/'
                                                                                             }">
                                                                    YouTube TV
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                </ul>
                                            </div>
                                        
                                    
                                        
                                            <div class="nav-brands__group nav-brands__group--x4">
                                                <h3 class="nav-brands__group__title glue-headline glue-headline--headline-6">
                                                    <a class="glue-header__link glue-font-weight-bold"
                                                       href="/brands#made_by_google"
                                                       data-analytics-on-click="{
                                                                                'eventCategory': 'global',
                                                                                'eventAction': 'nav submenu title click',
                                                                                'eventLabel': 'brands'
                                                                                }"
                                                       data-ga4-analytics-event="{
                                                                                 'event': 'nav_click',
                                                                                 'module_name': 'main nav',
                                                                                 'link_text': 'Made by Google',
                                                                                 'link_url': '/brands#made_by_google'
                                                                                 }">
                                                        Made by Google
                                                    </a>
                                                </h3>
                                                <ul class="nav-brands__group__list glue-no-bullet">
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/fitbit/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Fitbit'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Fitbit',
                                                                                             'link_url': '/brands/fitbit/overview/'
                                                                                             }">
                                                                    Fitbit
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-nest/overview/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Nest'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Nest',
                                                                                             'link_url': '/brands/google-nest/overview/overview/'
                                                                                             }">
                                                                    Google Nest
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-nest-display/nest-hub-7-inch/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Nest Display'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Nest Display',
                                                                                             'link_url': '/brands/google-nest-display/nest-hub-7-inch/overview/'
                                                                                             }">
                                                                    Google Nest Display
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-nest-pro/nest-pro-overview-1/brand-introduction/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Nest Pro'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Nest Pro',
                                                                                             'link_url': '/brands/google-nest-pro/nest-pro-overview-1/brand-introduction/'
                                                                                             }">
                                                                    Google Nest Pro
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-nest-safety/nest-protect/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Nest Safety'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Nest Safety',
                                                                                             'link_url': '/brands/google-nest-safety/nest-protect/overview/'
                                                                                             }">
                                                                    Google Nest Safety
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-nest-security/nest-cam-outdoor-wired-2nd-gen/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Nest Security'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Nest Security',
                                                                                             'link_url': '/brands/google-nest-security/nest-cam-outdoor-wired-2nd-gen/overview/'
                                                                                             }">
                                                                    Google Nest Security
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-nest-speakers/nest-audio/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Nest Speakers'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Nest Speakers',
                                                                                             'link_url': '/brands/google-nest-speakers/nest-audio/overview/'
                                                                                             }">
                                                                    Google Nest Speakers
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-nest-thermostat/nest-learning-thermostat-4th-gen/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Nest Thermostat'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Nest Thermostat',
                                                                                             'link_url': '/brands/google-nest-thermostat/nest-learning-thermostat-4th-gen/overview/'
                                                                                             }">
                                                                    Google Nest Thermostat
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-pixel/pixel-10a/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Pixel'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Pixel',
                                                                                             'link_url': '/brands/google-pixel/pixel-10a/overview/'
                                                                                             }">
                                                                    Google Pixel
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-pixelbook/pixelbook/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Pixelbook'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Pixelbook',
                                                                                             'link_url': '/brands/google-pixelbook/pixelbook/overview/'
                                                                                             }">
                                                                    Google Pixelbook
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-pixel-tablet/pixel-tablet/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Pixel Tablet'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Pixel Tablet',
                                                                                             'link_url': '/brands/google-pixel-tablet/pixel-tablet/overview/'
                                                                                             }">
                                                                    Google Pixel Tablet
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-pixel-wearables/pixel-watch-4/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Pixel Wearables'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Pixel Wearables',
                                                                                             'link_url': '/brands/google-pixel-wearables/pixel-watch-4/overview/'
                                                                                             }">
                                                                    Google Pixel Wearables
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-tv-streamer/google-tv-streamer/google-tv-streamer-4k/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google TV Streamer'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google TV Streamer',
                                                                                             'link_url': '/brands/google-tv-streamer/google-tv-streamer/google-tv-streamer-4k/overview/'
                                                                                             }">
                                                                    Google TV Streamer
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                        
                                                            <li class="nav-brands__group__item">
                                                                <a class="nav-brands__group__link glue-header__link" href="/brands/google-wifi/nest-wifi-pro/overview/"
                                                                   data-analytics-on-click="{
                                                                                            'eventCategory': 'global',
                                                                                            'eventAction': 'nav submenu link click',
                                                                                            'eventLabel': 'Google Wifi'
                                                                                            }"
                                                                   data-ga4-analytics-event="{
                                                                                             'event': 'nav_click',
                                                                                             'module_name': 'main nav',
                                                                                             'link_text': 'Google Wifi',
                                                                                             'link_url': '/brands/google-wifi/nest-wifi-pro/overview/'
                                                                                             }">
                                                                    Google Wifi
                                                                </a>
                                                            </li>
                                                        
                                                    
                                                </ul>
                                            </div>
                                        
                                    
                                </div>
                                <div class="nav-brands__footer">
                                    <a class="nav-brands__footer__link" href="/brands/" data-analytics-on-click="{
                                                                                                                 'eventCategory': 'global',
                                                                                                                 'eventAction': 'nav submenu cta link click',
                                                                                                                 'eventLabel': 'see all'
                                                                                                                 }"
                                       data-ga4-analytics-event="{
                                                                 'event': 'nav_click',
                                                                 'module_name': 'main nav',
                                                                 'link_text': 'See all',
                                                                 'link_url': '/brands/'
                                                                 }">
                                        See all
                                    </a>
                                </div>
                            </div>
                        </div>
                    
                </div>
            
        </div>

        <div class="glue-header__drawer-backdrop"></div>
    </header>


        

        
            <div class="notifications" data-component="notifications"></div>

        

        
            <hds-alert-manager class="alert-popups" data-component="alert-popups">
</hds-alert-manager>

        

        <main id="page-content" class="main">
            
    
    
    
    
    



<div class="brand" data-component="brand-page" data-brand-page-id="763">
    <div class="brand__header pmh-padding-3-top pmh-padding-3-bottom">
        <div class="glue-grid">
            <div class="glue-grid__col glue-grid__col--span-3-xl glue-grid__col--span-3-lg"></div>
            <div
                class="glue-grid__col glue-grid__col--span-9-xl glue-grid__col--span-9-lg glue-grid__col--span-12-md glue-grid__col--span-4-sm">
                <div class="glue-page pmh-page--brand">
                    <nav class="glue-breadcrumbs brand__breadcrumbs pmh-hide--mobile"
                         aria-label="You are here.">
                        <ol class="glue-breadcrumbs__list">
                            
                                <li class="glue-breadcrumbs__item" aria-level="1">
                                    <a class="glue-breadcrumbs__link" href="/en-us/homepage/brands/"
                                       data-analytics-on-click="{
                                                                'componentName': 'breadcrumbs',
                                                                'eventAction': 'breadcrumbs click',
                                                                'eventLabel': 'Brands'
                                                                }"
                                       data-ga4-analytics-event="{
                                                                 'event': 'nav_click',
                                                                 'module_name': 'breadcrumbs',
                                                                 'link_text': 'Brands',
                                                                 'link_url': '/en-us/homepage/brands/'
                                                                 }">
                                        Brands
                                    </a>
                                    <svg aria-hidden="true" class="glue-icon glue-icon--breadcrumb">
                                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="m7.59 18.59 1.41 1.41 8-8-8-8-1.41 1.41 6.58 6.59"/></svg>
                                    </svg>
                                </li>
                            
                            
                                
                            
                                
                                    <li class="glue-breadcrumbs__item
                                               "
                                         aria-level="3" >
                                            
                                                <span class="pmh-breadcrumbs__text">Google News</span>
                                            
                                            
                                                <svg aria-hidden="true" class="glue-icon glue-icon--breadcrumb">
                                                    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="m7.59 18.59 1.41 1.41 8-8-8-8-1.41 1.41 6.58 6.59"/></svg>
                                                </svg>
                                            
                                        </li>
                                
                            
                                
                                    <li class="glue-breadcrumbs__item
                                               "
                                         aria-level="4" >
                                            
                                                <span class="pmh-breadcrumbs__text">Use cases</span>
                                            
                                            
                                                <svg aria-hidden="true" class="glue-icon glue-icon--breadcrumb">
                                                    <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="m7.59 18.59 1.41 1.41 8-8-8-8-1.41 1.41 6.58 6.59"/></svg>
                                                </svg>
                                            
                                        </li>
                                
                            
                                
                                    <li class="glue-breadcrumbs__item
                                               glue-breadcrumbs__item--active"
                                         aria-level="5" >
                                            
                                                Press and media
                                            
                                            
                                        </li>
                                
                            
                        </ol>
                    </nav>

                    <div class="brand__header__mobile pmh-hide--desktop" data-component="brand-header"
                         data-brand-page-id="763">
                        <div class="glue-form__element glue-form__element--label-float">
                            <div class="mdc-select brand-select mdc-select--outlined mdc-select--nolabel glue-select"
                                 id="select-brand-select" data-component="mdc-select" data-ga4-analytics-event="{ 'event': 'dropdown_' }">
                                <input type="hidden" name="brand-select" value="/brands/google-news/overview/brand-introduction/"
                                       class="mdc-select--hidden-input" />
                                <div class="mdc-select__anchor" role="button" aria-haspopup="listbox"
                                     aria-labelledby="label-brand-select selectedtext-brand-select">
                                    <span class="glue-form__floating-label">
                                        Brands
                                    </span>
                                    <span id="label-brand-select" class="glue-visually-hidden">
                                        Select a brand
                                    </span>
                                    <span id="selectedtext-brand-select" class="mdc-select__selected-text">
                                        Google News
                                    </span>
                                    <span class="mdc-select__dropdown-icon">
                                        <svg class="mdc-select__dropdown-icon-graphic" viewBox="7 10 10 5" xmlns="http://www.w3.org/2000/svg"><path class="mdc-select__dropdown-icon-inactive" fill-rule="evenodd" d="M7 10l5 5 5-5z"/><path class="mdc-select__dropdown-icon-active" fill-rule="evenodd" d="M7 15l5-5 5 5z"/></svg>
                                    </span>
                                    <span class="mdc-notched-outline">
                                        <span class="mdc-notched-outline__leading"></span>
                                        <span class="mdc-notched-outline__trailing"></span>
                                    </span>
                                </div>
                                <div class="mdc-select__menu mdc-menu mdc-menu-surface" role="listbox">
                                    <ul class="mdc-list">
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-1"
                                                data-value="/brands/android/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="1" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Android',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Android'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Android
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-9"
                                                data-value="/brands/android-auto/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="2" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Android Auto',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Android Auto'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Android Auto
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-6"
                                                data-value="/brands/android-enterprise/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="3" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Android Enterprise',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Android Enterprise'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Android Enterprise
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-7"
                                                data-value="/brands/android-tv/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="4" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Android TV',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Android TV'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Android TV
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-131"
                                                data-value="/brands/android-xr/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="5" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Android XR',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Android XR'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Android XR
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-90"
                                                data-value="/brands/cars-with-google-built-in/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="6" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Cars with Google built\u002Din',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Cars with Google built\u002Din'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Cars with Google built-in
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-68"
                                                data-value="/brands/chrome/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="7" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Chrome',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Chrome'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Chrome
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-11"
                                                data-value="/brands/chromebook/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="8" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Chromebook',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Chromebook'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Chromebook
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-43"
                                                data-value="/brands/diversity-inclusion/diversity-equity-inclusion/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="9" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Diversity and inclusion',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Diversity and inclusion'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Diversity and inclusion
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-34"
                                                data-value="/brands/family-link/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="10" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Family Link',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Family Link'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Family Link
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-74"
                                                data-value="/brands/fast-pair/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="11" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Fast Pair',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Fast Pair'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Fast Pair
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-132"
                                                data-value="/brands/find-hub/find-hub/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="12" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Find Hub',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Find Hub'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Find Hub
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-50"
                                                data-value="/brands/fitbit/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="13" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Fitbit',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Fitbit'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Fitbit
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-77"
                                                data-value="/brands/google/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="14" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-27"
                                                data-value="/brands/google-assistant/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="15" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Assistant',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Assistant'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Assistant
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-26"
                                                data-value="/brands/google-cast/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="16" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Cast',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Cast'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Cast
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-115"
                                                data-value="/brands/google-gemini/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="17" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Gemini',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Gemini'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Gemini
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-39"
                                                data-value="/brands/google-lens/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="18" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Lens',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Lens'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Lens
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-45"
                                                data-value="/brands/google-maps/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="19" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Maps',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Maps'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Maps
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-60"
                                                data-value="/brands/google-meet/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="20" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Meet',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Meet'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Meet
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-85"
                                                data-value="/brands/google-messages/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="21" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Messages',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Messages'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Messages
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-20"
                                                data-value="/brands/google-nest/overview/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="22" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Nest',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Nest'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Nest
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-93"
                                                data-value="/brands/google-nest-display/nest-hub-7-inch/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="23" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Nest Display',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Nest Display'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Nest Display
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-95"
                                                data-value="/brands/google-nest-pro/nest-pro-overview-1/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="24" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Nest Pro',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Nest Pro'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Nest Pro
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-96"
                                                data-value="/brands/google-nest-safety/nest-protect/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="25" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Nest Safety',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Nest Safety'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Nest Safety
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-108"
                                                data-value="/brands/google-nest-security/nest-cam-outdoor-wired-2nd-gen/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="26" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Nest Security',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Nest Security'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Nest Security
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-98"
                                                data-value="/brands/google-nest-speakers/nest-audio/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="27" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Nest Speakers',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Nest Speakers'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Nest Speakers
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-92"
                                                data-value="/brands/google-nest-thermostat/nest-learning-thermostat-4th-gen/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="28" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Nest Thermostat',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Nest Thermostat'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Nest Thermostat
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item  mdc-list-item--selected "
                                                id="brand-select-29"
                                                data-value="/brands/google-news/overview/brand-introduction/" role="option"
                                                aria-selected="true"
                                                 aria-setsize="51"
                                                aria-posinset="29" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google News',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google News'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google News
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-16"
                                                data-value="/brands/google-one/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="30" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google One',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google One'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google One
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-145"
                                                data-value="/brands/google-pay-google-wallet/how-our-brands-work-together/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="31" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Pay \u0026 Google Wallet',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Pay \u0026 Google Wallet'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Pay &amp; Google Wallet
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-30"
                                                data-value="/brands/google-photos/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="32" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Photos',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Photos'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Photos
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-23"
                                                data-value="/brands/google-pixel/pixel-10a/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="33" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Pixel',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Pixel'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Pixel
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-99"
                                                data-value="/brands/google-pixelbook/pixelbook/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="34" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Pixelbook',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Pixelbook'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Pixelbook
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-114"
                                                data-value="/brands/google-pixel-tablet/pixel-tablet/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="35" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Pixel Tablet',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Pixel Tablet'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Pixel Tablet
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-103"
                                                data-value="/brands/google-pixel-wearables/pixel-watch-4/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="36" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Pixel Wearables',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Pixel Wearables'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Pixel Wearables
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-2"
                                                data-value="/brands/google-play/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="37" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Play',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Play'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Play
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-31"
                                                data-value="/brands/google-play-protect/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="38" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Play Protect',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Play Protect'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Play Protect
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-44"
                                                data-value="/brands/google-tv/overview/google-tv-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="39" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google TV',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google TV'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google TV
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-12"
                                                data-value="/brands/google-tv-streamer/google-tv-streamer/google-tv-streamer-4k/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="40" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google TV Streamer',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google TV Streamer'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google TV Streamer
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-106"
                                                data-value="/brands/google-wifi/nest-wifi-pro/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="41" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Google Wifi',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Google Wifi'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Google Wifi
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-59"
                                                data-value="/brands/nd4c-guidance/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="42" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - ND4C Guidance',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'ND4C Guidance'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    ND4C Guidance
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-133"
                                                data-value="/brands/quick-share/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="43" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Quick Share',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Quick Share'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Quick Share
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-128"
                                                data-value="/brands/waze/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="44" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Waze',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Waze'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Waze
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-8"
                                                data-value="/brands/wear-os-by-google/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="45" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Wear OS by Google',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Wear OS by Google'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Wear OS by Google
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-56"
                                                data-value="/brands/works-with-google-home/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="46" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - Works With Google Home',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'Works With Google Home'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    Works With Google Home
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-88"
                                                data-value="/brands/youtube/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="47" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - YouTube',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'YouTube'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    YouTube
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-37"
                                                data-value="/brands/youtube-kids/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="48" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - YouTube Kids',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'YouTube Kids'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    YouTube Kids
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-32"
                                                data-value="/brands/youtube-music/overview/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="49" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - YouTube Music',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'YouTube Music'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    YouTube Music
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-4"
                                                data-value="/brands/youtube-premium/introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="50" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - YouTube Premium',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'YouTube Premium'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    YouTube Premium
                                                </span>
                                            </li>
                                        
                                            <li class="mdc-list-item "
                                                id="brand-select-33"
                                                data-value="/brands/youtube-tv/overview/brand-introduction/" role="option"
                                                 aria-setsize="51"
                                                aria-posinset="51" data-analytics-on-click="{
                                                                                                               'eventAction': 'brand mobile header - select',
                                                                                                               'eventLabel': 'brand - YouTube TV',
                                                                                                               'componentName': 'brand mobile header'
                                                                                                               }" data-ga4-analytics-event="{
                                                                                                                                 'event': 'dropdown_select',
                                                                                                                                 'link_text': 'YouTube TV'
                                                                                                                                 }">
                                                <span class="mdc-list-item__ripple"></span>
                                                <span class="mdc-list-item__text">
                                                    YouTube TV
                                                </span>
                                            </li>
                                        
                                    </ul>
                                </div>
                            </div>
                        </div>
                        <div class="glue-form__element glue-form__element--label-float">
                            <div class="mdc-select page-select mdc-select--outlined mdc-select--nolabel glue-select"
                                 id="select-page-select" data-component="mdc-select" data-ga4-analytics-event="{ 'event': 'dropdown_' }">
                                <input type="hidden" name="page-select" value="/brands/google-news/use-cases/press-and-media/"
                                       class="mdc-select--hidden-input" />
                                <div class="mdc-select__anchor" role="button" aria-haspopup="listbox"
                                     aria-labelledby="label-page-select selectedtext-page-select">
                                    <span class="glue-form__floating-label">
                                        Page selection
                                    </span>
                                    <span id="label-page-select" class="glue-visually-hidden">
                                        Select a page
                                    </span>
                                    <span id="selectedtext-page-select" class="mdc-select__selected-text">
                                        Press and media
                                    </span>
                                    <span class="mdc-select__dropdown-icon">
                                        <svg class="mdc-select__dropdown-icon-graphic" viewBox="7 10 10 5" xmlns="http://www.w3.org/2000/svg"><path class="mdc-select__dropdown-icon-inactive" fill-rule="evenodd" d="M7 10l5 5 5-5z"/><path class="mdc-select__dropdown-icon-active" fill-rule="evenodd" d="M7 15l5-5 5 5z"/></svg>
                                    </span>
                                    <span class="mdc-notched-outline">
                                        <span class="mdc-notched-outline__leading"></span>
                                        <span class="mdc-notched-outline__trailing"></span>
                                    </span>
                                </div>
                                <div class="mdc-select__menu mdc-menu mdc-menu-surface" role="listbox">
                                    <ul class="mdc-list">
                                        <li class="mdc-list-item mdc-list-item--selected" data-value="/brands/google-news/use-cases/press-and-media/"
                                            role="option" aria-setsize="1" aria-posinset="1" aria-selected="true"
                                            data-ga4-analytics-event="{
                                                                      'event': 'dropdown_select',
                                                                      'link_text': 'Press and media'
                                                                      }">
                                            <span class="mdc-list-item__ripple"></span>
                                            <span class="mdc-list-item__text">Press and media</span>
                                        </li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>

                    
                    
                    
                    
                        <div class="brand__header__logo__container pmh-hide--mobile">
                            <picture>
                                
                                    
                                    <source data-srcset="https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/images/googlenews_breadcrumb.max-300x70.format-webp.webp?X-Goog-Algorithm=GOOG4-RSA-SHA256&amp;X-Goog-Credential=wagtail%40pe-portal-consumer-prod.iam.gserviceaccount.com%2F20260510%2Fauto%2Fstorage%2Fgoog4_request&amp;X-Goog-Date=20260510T101706Z&amp;X-Goog-Expires=86400&amp;X-Goog-SignedHeaders=host&amp;X-Goog-Signature=857816952f489e4ae53c2c7c5abde440680875c7ffce5f84784f324b73a13d509e35d116cb6bcb2286fa80ca568c9a1fe9a0911553271d690a4833934bb805e89a2bfef354fd7537bd6b8d5eacbdd09bd1e76e97b752c78bbd1b7288ee8180c98e44a52002251a705a9c19e187ad424d7c243f785531651d715adcd164a60c6e3334082a232ebf5ecb037f226674fa68cfe5d79179159aa2357a13dec2ffc0802203fcba1bf6f40cf61fa050fbce45b39063547be74213c49b671bd2e967abd260fe57546303b1508b8008c559767d2833ac329c05a9c8de26f12b2d382168fe469d329dfc5ad50e25b4c9d18b999683c2ab40593da4fe42186c0c2187f3439d" type="image/webp" />
                                    <source data-srcset="https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/images/googlenews_breadcrumb.max-300x70.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&amp;X-Goog-Credential=wagtail%40pe-portal-consumer-prod.iam.gserviceaccount.com%2F20260510%2Fauto%2Fstorage%2Fgoog4_request&amp;X-Goog-Date=20260510T101706Z&amp;X-Goog-Expires=86400&amp;X-Goog-SignedHeaders=host&amp;X-Goog-Signature=79f887d131c04a0ba68b0927dc9e312469d657837d24b8f3b2316dc801a0c606efb125699f64ca5789b9bd949fdd7c05a03f5752eab455e1306e6bce3e1bc38cf9f21b9befb7ece524df5a7ed8d98d8cfa37670492829654fa67ea1bb393dab1c49a267d436c58ab0d77649df8f42e361f0fbc515a5e86b09c26c3ce70addd183d00184c4d2c1015ec3a1e56a275f17db1fc3f4cec918f477f5e6027b94f9f24f40305c8117455fe65cdaf8c11f82188b7ae03db4623d514a85fb6456484019f97d728e312d1827fe4698067bb3210c792d2bacf3554a5f07e1ccb50eee2857fbcce4b42b5784d282e9981cf5d34a73c86edee86245967cd28f0b97bc438861a" />
                                
                                <img title=" logo"
                                     alt="googlenews_breadcrumb.png"
                                     class="brand__header__logo brand__header__logo--folder" width="300"
                                     height="46" data-src="https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/images/googlenews_breadcrumb.max-300x70.png?X-Goog-Algorithm=GOOG4-RSA-SHA256&amp;X-Goog-Credential=wagtail%40pe-portal-consumer-prod.iam.gserviceaccount.com%2F20260510%2Fauto%2Fstorage%2Fgoog4_request&amp;X-Goog-Date=20260510T101706Z&amp;X-Goog-Expires=86400&amp;X-Goog-SignedHeaders=host&amp;X-Goog-Signature=79f887d131c04a0ba68b0927dc9e312469d657837d24b8f3b2316dc801a0c606efb125699f64ca5789b9bd949fdd7c05a03f5752eab455e1306e6bce3e1bc38cf9f21b9befb7ece524df5a7ed8d98d8cfa37670492829654fa67ea1bb393dab1c49a267d436c58ab0d77649df8f42e361f0fbc515a5e86b09c26c3ce70addd183d00184c4d2c1015ec3a1e56a275f17db1fc3f4cec918f477f5e6027b94f9f24f40305c8117455fe65cdaf8c11f82188b7ae03db4623d514a85fb6456484019f97d728e312d1827fe4698067bb3210c792d2bacf3554a5f07e1ccb50eee2857fbcce4b42b5784d282e9981cf5d34a73c86edee86245967cd28f0b97bc438861a" />
                            </picture>
                        </div>
                    
                </div>
            </div>
        </div>
    </div>

    <div class="brand__main pmh-padding-3-top pmh-padding-3-bottom">
        <div class="glue-grid">
            <nav class="brand__main__sidebar glue-grid__col glue-grid__col--span-3-xl glue-grid__col--span-3-lg pmh-hide--mobile"
                 aria-label="Brands navigation">
                <div class="brand_tree__container" data-component="brand-tree">
                    <ul class="brand__tree glue-no-bullet pmh-margin-2-right">
                        <li class="brand-tree__placeholder-container">
                            
                                <div class="brand__tree_dummy--wrapper">
                                    <div class="brand__tree__dummy">
                                        <div class="brand__tree__dummy--bullet"></div>
                                        <div class="brand__tree__dummy--text"></div>
                                        <div class="placeholder__shimmer animated-background"></div>

                                    </div>
                                </div>
                            
                                <div class="brand__tree_dummy--wrapper">
                                    <div class="brand__tree__dummy">
                                        <div class="brand__tree__dummy--bullet"></div>
                                        <div class="brand__tree__dummy--text"></div>
                                        <div class="placeholder__shimmer animated-background"></div>

                                    </div>
                                </div>
                            
                                <div class="brand__tree_dummy--wrapper">
                                    <div class="brand__tree__dummy">
                                        <div class="brand__tree__dummy--bullet"></div>
                                        <div class="brand__tree__dummy--text"></div>
                                        <div class="placeholder__shimmer animated-background"></div>

                                    </div>
                                </div>
                            
                                <div class="brand__tree_dummy--wrapper">
                                    <div class="brand__tree__dummy">
                                        <div class="brand__tree__dummy--bullet"></div>
                                        <div class="brand__tree__dummy--text"></div>
                                        <div class="placeholder__shimmer animated-background"></div>

                                    </div>
                                </div>
                            
                                <div class="brand__tree_dummy--wrapper">
                                    <div class="brand__tree__dummy">
                                        <div class="brand__tree__dummy--bullet"></div>
                                        <div class="brand__tree__dummy--text"></div>
                                        <div class="placeholder__shimmer animated-background"></div>

                                    </div>
                                </div>
                            
                                <div class="brand__tree_dummy--wrapper">
                                    <div class="brand__tree__dummy">
                                        <div class="brand__tree__dummy--bullet"></div>
                                        <div class="brand__tree__dummy--text"></div>
                                        <div class="placeholder__shimmer animated-background"></div>

                                    </div>
                                </div>
                            
                        </li>
                    </ul>
                </div>
            </nav>
            <div
                class="brand__main__content glue-grid__col glue-grid__col--span-9-xl glue-grid__col--span-9-lg glue-grid__col--span-12-md glue-grid__col--span-4-sm">
                <div class="glue-page pmh-page--brand">
                    
                        <div class="brand__language">
                            







    <div class="pmh-static-lang">
        <i class="material-icons glue-caption pmh-margin-1-right" aria-hidden="true">
            language
        </i>
        English - United States
    </div>
    



                        </div>

                        <div class="blocks__container">
                            
                                <div class="pmh-block block-section block-section--title"
                                     id="block-section-title-f618a9be-20e5-4123-b633-6f6d0a533cd5"
                                     data-component-name="title">
                                    

<div class="block-title">
    <h1 class="glue-headline glue-headline--headline-2 glue-spacer-2-bottom" id="press-and-media">
        Press and media
    </h1>
    
</div>



                                </div>
                            
                                <div class="pmh-block block-section block-section--copy"
                                     id="block-section-copy-25bfcb45-5056-4e32-be8f-810e78be019d"
                                     data-component-name="copy">
                                    

<div class="block-copy">
    <p>Any press releases mentioning the Google News brand or products requires review and approval. Please send materials to <a href="mailto:press@google.com" target="_blank">press@google.com</a> for review at least 72 hours before you plan to issue it, and include any questions you may have.</p><p>Google does not generally provide quotes for press materials. This can be reviewed on a case-by-case basis.</p><p>Please do not include the Google boilerplate, contact information or Google logo on your press release. These details imply a joint-press release, and Google does not support joint-releases. We appreciate your support and cooperation.</p>
</div>

                                </div>
                            
                        </div>
                    
                </div>

                

<div class="pmh-back-to-top glue-text-center">
    <a class="glue-button glue-button--low-emphasis glue-button--icon-right" href="#page-top" data-analytics-on-click="{
        'eventCategory': 'global',
        'eventAction': 'back to the top - cta click',
        'eventLabel': 'back to the top'
        }" data-ga4-analytics-event="{
        'event': 'anchor_link_click',
        'link_text': 'Back to top',
        'link_url': '#page-top'
        }">
        Back to top
        <svg aria-hidden="true" class="glue-icon">
            <svg viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg"><path d="m9 1.5c-4.2 0-7.5 3.3-7.5 7.5s3.3 7.5 7.5 7.5 7.5-3.3 7.5-7.5-3.3-7.5-7.5-7.5zm0 13-1-1 3.8-3.8h-8.3v-1.4h8.4l-3.8-3.8.9-1 5.5 5.5z"/></svg>
        </svg>
    </a>
</div>

            </div>
        </div>
    </div>
</div>


    

            
            
        </main>

        
            

<footer class="glue-footer glue-spacer-5-top pmh-footer" data-component="footer">
    <section class="glue-footer__site-links" aria-label="Footer navigation">
        <div class="glue-footer__site-links-grid glue-footer__site-links-grid--3-col">
            <div class="glue-footer__site-links-column" role="region" id="footer-links-column-1"
                 aria-labelledby="footer-links-column-1-title">
                <span class="glue-footer__site-links-header" id="footer-links-column-1-title" data-analytics-on-click="{
                                                                                                                       'eventCategory': 'global',
                                                                                                                       'eventAction': 'footer accordion click',
                                                                                                                       'eventLabel': 'partner marketing hub'
                                                                                                                       }">
                    <span class="glue-footer__site-links-header-button" id="footer-links-column-1-toggle"
                          data-ga4-analytics-event="{
                                                    'accordion_header': 'Partner Marketing Hub'
                                                    }">
                        <span class="glue-footer__site-links-header-text">
                            Partner Marketing Hub
                        </span>
                        <svg aria-hidden="true" class="glue-icon glue-icon--footer glue-expansion-panel__header-arrow">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M5.41 7.59L4 9l8 8 8-8-1.41-1.41L12 14.17"></path></svg>

                        </svg>
                    </span>
                </span>
                <div class="glue-footer__site-links-list" id="footer-links-column-1-content">
                    <ul class="glue-no-bullet">
                        <li class="glue-footer__site-links-list-item">
                            <a class="glue-footer__link" href="/brands/" data-analytics-on-click="{
                                                                                                  'eventCategory': 'global',
                                                                                                  'eventAction': 'footer menu link click',
                                                                                                  'eventLabel': 'brands'
                                                                                                  }"
                               data-ga4-analytics-event="{
                                                         'event': 'nav_click',
                                                         'module_name': 'footer',
                                                         'link_text': 'Brands',
                                                         'link_url': '/brands/'
                                                         }">
                                Brands
                            </a>
                        </li>
                        <li class="glue-footer__site-links-list-item">
                            <a class="glue-footer__link" href="/marketing-opportunities/" data-analytics-on-click="{
                                                                                                                   'eventCategory': 'global',
                                                                                                                   'eventAction': 'footer menu link click',
                                                                                                                   'eventLabel': 'marketing opportunities'
                                                                                                                   }"
                               data-ga4-analytics-event="{
                                                         'event': 'nav_click',
                                                         'module_name': 'footer',
                                                         'link_text': 'Marketing opportunities',
                                                         'link_url': '/marketing-opportunities/'
                                                         }">
                                Marketing opportunities
                            </a>
                        </li>
                        <li class="glue-footer__site-links-list-item">
                            <a class="glue-footer__link" href="/tools/" data-analytics-on-click="{
                                                                                                 'eventCategory': 'global',
                                                                                                 'eventAction': 'footer menu link click',
                                                                                                 'eventLabel': 'tools'
                                                                                                 }"
                               data-ga4-analytics-event="{
                                                         'event': 'nav_click',
                                                         'module_name': 'footer',
                                                         'link_text': 'Tools',
                                                         'link_url': '/tools/'
                                                         }">
                                Tools
                            </a>
                        </li>
                        <li class="glue-footer__site-links-list-item">
                            <a class="glue-footer__link" href="/asset-approval/" data-analytics-on-click="{
                                                                                                          'eventCategory': 'global',
                                                                                                          'eventAction': 'footer menu link click',
                                                                                                          'eventLabel': 'asset approval'
                                                                                                          }"
                               data-ga4-analytics-event="{
                                                         'event': 'nav_click',
                                                         'module_name': 'footer',
                                                         'link_text': 'Asset approval',
                                                         'link_url': '/asset-approval/'
                                                         }">
                                Asset approval
                            </a>
                        </li>
                    </ul>
                </div>
            </div>
            <div class="glue-footer__site-links-column" role="region" id="footer-links-column-2"
                 aria-labelledby="footer-links-column-2-title">
                <span class="glue-footer__site-links-header" id="footer-links-column-2-title" data-analytics-on-click="{
                                                                                                                       'eventCategory': 'global',
                                                                                                                       'eventAction': 'footer accordion click',
                                                                                                                       'eventLabel': 'support'
                                                                                                                       }">
                    <span class="glue-footer__site-links-header-button" id="footer-links-column-2-toggle"
                          data-ga4-analytics-event="{
                                                    'accordion_header': 'Support'
                                                    }">
                        <span class="glue-footer__site-links-header-text">
                            Support
                        </span>
                        <svg aria-hidden="true" class="glue-icon glue-icon--footer glue-expansion-panel__header-arrow">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M5.41 7.59L4 9l8 8 8-8-1.41-1.41L12 14.17"></path></svg>

                        </svg>
                    </span>
                </span>
                <div class="glue-footer__site-links-list" id="footer-links-column-2-content">
                    <ul class="glue-no-bullet">
                        <li class="glue-footer__site-links-list-item">
                            <a data-analytics-on-click="{
                                                        'eventCategory': 'global',
                                                        'eventAction': 'footer menu link click',
                                                        'eventLabel': 'contact us'
                                                        }" data-ga4-analytics-event="{
                                                                                                                    'event': 'nav_click',
                                                                                                                    'module_name': 'footer',
                                                                                                                    'link_text': 'Contact us',
                                                                                                                    'link_url': '/support/contact-us/'
                                                                                                                    }"
                               class="glue-footer__link" href="/support/contact-us/" rel="noopener noreferrer">
                                Contact us
                            </a>
                        </li>
                        <li class="glue-footer__site-links-list-item">
                            <a data-analytics-on-click="{
                                                        'eventCategory': 'global',
                                                        'eventAction': 'footer menu link click',
                                                        'eventLabel': 'faq'
                                                        }" data-ga4-analytics-event="{
                                                                                                                    'event': 'nav_click',
                                                                                                                    'module_name': 'footer',
                                                                                                                    'link_text': 'FAQ',
                                                                                                                    'link_url': '/support/faq/'
                                                                                                                    }"
                               class="glue-footer__link" href="/support/faq/">
                                FAQ
                            </a>
                        </li>
                    </ul>
                </div>
            </div>
            <div class="glue-footer__site-links-column" role="region" id="footer-links-column-3"
                 aria-labelledby="footer-links-column-3-title">
                <span class="glue-footer__site-links-header" id="footer-links-column-3-title" data-analytics-on-click="{
                                                                                                                       'eventCategory': 'global',
                                                                                                                       'eventAction': 'footer accordion click',
                                                                                                                       'eventLabel': 'profile'
                                                                                                                       }">
                    <span class="glue-footer__site-links-header-button" id="footer-links-column-3-toggle"
                          data-ga4-analytics-event="{
                                                    'accordion_header': 'Profile'
                                                    }">
                        <span class="glue-footer__site-links-header-text">
                            Profile
                        </span>
                        <svg aria-hidden="true" class="glue-icon glue-icon--footer glue-expansion-panel__header-arrow">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M5.41 7.59L4 9l8 8 8-8-1.41-1.41L12 14.17"></path></svg>

                        </svg>
                    </span>
                </span>
                <div class="glue-footer__site-links-list" id="footer-links-column-3-content">
                    <ul class="glue-no-bullet">
                        <li class="glue-footer__site-links-list-item">
                            <a class="glue-footer__link" href="/profile/" data-analytics-on-click="{
                                                                                                             'eventCategory': 'global',
                                                                                                             'eventAction': 'footer menu link click',
                                                                                                             'eventLabel': 'my account'
                                                                                                             }"
                               data-ga4-analytics-event="{
                                                         'event': 'nav_click',
                                                         'module_name': 'footer',
                                                         'link_text': 'My account',
                                                         'link_url': '/profile/'
                                                         }">
                                My account
                            </a>
                        </li>
                        <li class="glue-footer__site-links-list-item">
                            <a class="glue-footer__link" href="/notifications/"
                               data-analytics-on-click="{
                                                        'eventCategory': 'global',
                                                        'eventAction': 'footer menu link click',
                                                        'eventLabel': 'notifications'
                                                        }" data-ga4-analytics-event="{
                                                                                                                    'event': 'nav_click',
                                                                                                                    'module_name': 'footer',
                                                                                                                    'link_text': 'Notifications',
                                                                                                                    'link_url': '/notifications/'
                                                                                                                    }">
                                Notifications
                            </a>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    </section>

    <section class="glue-footer__global">
        <div class="glue-footer__logo">
            <a class="glue-footer__link" href="https://www.google.com" title="Google"
               aria-label="Google link with logo black and white" target="_blank"
               rel="noopener noreferrer" data-analytics-on-click="{
                                                                  'eventCategory': 'global',
                                                                  'eventAction': 'footer logo click',
                                                                  'eventLabel': 'logo'
                                                                  }" data-ga4-analytics-event="{
                                                                                                              'event': 'nav_click',
                                                                                                              'module_name': 'footer',
                                                                                                              'link_text': 'Google link with logo black and white',
                                                                                                              'link_url': 'https://www.google.com'
                                                                                                              }">
                <svg aria-hidden="true" class="glue-footer__logo-img">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 74 24"><path d="M9.24 8.19v2.46h5.88c-.18 1.38-.64 2.39-1.34 3.1-.86.86-2.2 1.8-4.54 1.8-3.62 0-6.45-2.92-6.45-6.54s2.83-6.54 6.45-6.54c1.95 0 3.38.77 4.43 1.76L15.4 2.5C13.94 1.08 11.98 0 9.24 0 4.28 0 .11 4.04.11 9s4.17 9 9.13 9c2.68 0 4.7-.88 6.28-2.52 1.62-1.62 2.13-3.91 2.13-5.75 0-.57-.04-1.1-.13-1.54H9.24zm15.76-2c-3.21 0-5.83 2.44-5.83 5.81 0 3.34 2.62 5.81 5.83 5.81s5.83-2.46 5.83-5.81c0-3.37-2.62-5.81-5.83-5.81zm0 9.33c-1.76 0-3.28-1.45-3.28-3.52 0-2.09 1.52-3.52 3.28-3.52s3.28 1.43 3.28 3.52c0 2.07-1.52 3.52-3.28 3.52zm28.58-8.03h-.09c-.57-.68-1.67-1.3-3.06-1.3C47.53 6.19 45 8.72 45 12c0 3.26 2.53 5.81 5.43 5.81 1.39 0 2.49-.62 3.06-1.32h.09v.81c0 2.22-1.19 3.41-3.1 3.41-1.56 0-2.53-1.12-2.93-2.07l-2.22.92c.64 1.54 2.33 3.43 5.15 3.43 2.99 0 5.52-1.76 5.52-6.05V6.49h-2.42v1zm-2.93 8.03c-1.76 0-3.1-1.5-3.1-3.52 0-2.05 1.34-3.52 3.1-3.52 1.74 0 3.1 1.5 3.1 3.54.01 2.03-1.36 3.5-3.1 3.5zM38 6.19c-3.21 0-5.83 2.44-5.83 5.81 0 3.34 2.62 5.81 5.83 5.81s5.83-2.46 5.83-5.81c0-3.37-2.62-5.81-5.83-5.81zm0 9.33c-1.76 0-3.28-1.45-3.28-3.52 0-2.09 1.52-3.52 3.28-3.52s3.28 1.43 3.28 3.52c0 2.07-1.52 3.52-3.28 3.52zM58 .24h2.51v17.57H58zm10.26 15.28c-1.3 0-2.22-.59-2.82-1.76l7.77-3.21-.26-.66c-.48-1.3-1.96-3.7-4.97-3.7-2.99 0-5.48 2.35-5.48 5.81 0 3.26 2.46 5.81 5.76 5.81 2.66 0 4.2-1.63 4.84-2.57l-1.98-1.32c-.66.96-1.56 1.6-2.86 1.6zm-.18-7.15c1.03 0 1.91.53 2.2 1.28l-5.25 2.17c0-2.44 1.73-3.45 3.05-3.45z"></path></svg>

                </svg>
            </a>
        </div>

        <ul class="glue-footer__global-links glue-no-bullet">
            <li class="glue-footer__global-links-list-item">
                <a class="glue-footer__link" href="https://about.google" target="_blank" rel="noopener noreferrer"
                   data-analytics-on-click="{
                                            'eventCategory': 'global',
                                            'eventAction': 'footer global link click',
                                            'eventLabel': 'about google'
                                            }" data-ga4-analytics-event="{
                                                                                            'event': 'nav_click',
                                                                                            'module_name': 'footer',
                                                                                            'link_text': 'About Google',
                                                                                            'link_url': 'https://about.google'
                                                                                            }">
                    About Google
                </a>
            </li>

            <li class="glue-footer__global-links-list-item">
                <a class="glue-footer__link" href="https://about.google/products/" target="_blank"
                   rel="noopener noreferrer" data-analytics-on-click="{
                                                                      'eventCategory': 'global',
                                                                      'eventAction': 'footer global link click',
                                                                      'eventLabel': 'google products'
                                                                      }" data-ga4-analytics-event="{
                                                                                                                      'event': 'nav_click',
                                                                                                                      'module_name': 'footer',
                                                                                                                      'link_text': 'Google products',
                                                                                                                      'link_url': 'https://about.google/products/'
                                                                                                                      }">
                    Google products
                </a>
            </li>

            <li class="glue-footer__global-links-list-item">
                <a class="glue-footer__link" href="https://policies.google.com/privacy" target="_blank"
                   rel="noopener noreferrer" data-analytics-on-click="{
                                                                      'eventCategory': 'global',
                                                                      'eventAction': 'footer global link click',
                                                                      'eventLabel': 'privacy'
                                                                      }" data-ga4-analytics-event="{
                                                                                                                      'event': 'nav_click',
                                                                                                                      'module_name': 'footer',
                                                                                                                      'link_text': 'Privacy',
                                                                                                                      'link_url': 'https://policies.google.com/privacy'
                                                                                                                      }">
                    Privacy
                </a>
            </li>

            <li class="glue-footer__global-links-list-item">
                <a class="glue-footer__link" href="https://policies.google.com/terms" target="_blank"
                   rel="noopener noreferrer" data-analytics-on-click="{
                                                                      'eventCategory': 'global',
                                                                      'eventAction': 'footer global link click',
                                                                      'eventLabel': 'terms'
                                                                      }" data-ga4-analytics-event="{
                                                                                                                      'event': 'nav_click',
                                                                                                                      'module_name': 'footer',
                                                                                                                      'link_text': 'Terms',
                                                                                                                      'link_url': 'https://policies.google.com/terms'
                                                                                                                      }">
                    Terms
                </a>
            </li>
        </ul>

        <ul class="glue-footer__global-links glue-footer__global-links--extra glue-no-bullet">
            <li class="glue-footer__global-links-list-item glue-footer__global-links-list-item--extra">
                <a class="glue-footer__link" href="https://support.google.com/?hl=en" target="_blank"
                   rel="noopener noreferrer" title="Help" data-analytics-on-click="{
                                                                                   'eventCategory': 'global',
                                                                                   'eventAction': 'footer support click',
                                                                                   'eventLabel': 'help'
                                                                                   }" data-ga4-analytics-event="{
                                                                                                                                   'event': 'nav_click',
                                                                                                                                   'module_name': 'footer',
                                                                                                                                   'link_text': 'Help',
                                                                                                                                   'link_url': 'https://support.google.com/?hl=en'
                                                                                                                                   }">
                    <svg aria-hidden="true" class="glue-icon glue-icon--24px glue-icon--footer-help">
                        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="m12 2c-5.52 0-10 4.48-10 10s4.48 10 10 10 10-4.48 10-10-4.48-10-10-10zm1 17h-2v-2h2zm2.07-7.75-.9.92c-.72.73-1.17 1.33-1.17 2.83h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2h-2c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z"/></svg>
                    </svg>
                    Help
                </a>
            </li>
        </ul>
    </section>
</footer>

        

        

<div class="pmh-snackbar" data-component="snackbar">
    <div class="mdc-snackbar">
        <div class="mdc-snackbar__surface">
            <div class="mdc-snackbar__label" role="status" aria-live="polite">
            </div>
            <div class="mdc-snackbar__actions">
                <button type="button" class="mdc-button mdc-snackbar__action pmh-hide" aria-hidden="true">
                    <div class="mdc-button__ripple"></div>
                    <span class="mdc-button__label"></span>
                </button>
                <button type="button" class="mdc-icon-button mdc-snackbar__dismiss material-icons"
                    aria-label="Close snackbar">
                    close
                </button>
            </div>
        </div>
    </div>
</div>

        <div class="session-closed-popup">
    <hds-dialog class="session-closed-popup__dialog" data-component="session-closed-popup-manager"
                block-backdrop="true">
        <div class="session-closed-popup__content">
            <p class="session-closed-popup__text">Your session has been closed</p>
            <hds-cta class="session-closed-popup__login-button hds-button-mte hds-button-mte--filled hds-button-mte--32"
                     href="/login/" label="Log in"></hds-cta>
        </div>
    </hds-dialog>
</div>

        

<template id="pmh-urls">
    <!-- NOTE: Add urls to be accessed in the app -->
    <!-- with <span data-urlname=""></span> -->
    <span data-tracker-url="/metrics/pulse/"></span>
</template>


    <!--Global scripts-->
        <script src="https://ajax.googleapis.com/ajax/libs/hammerjs/2.0.8/hammer.min.js"></script>
        <script src="https://www.gstatic.com/firebasejs/8.10.1/firebase-app.js" defer></script>
<script src="https://www.gstatic.com/firebasejs/8.10.1/firebase-auth.js" defer></script>

        

<link rel="preload" as="script" href="https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/js/hubdesignsystem.9c476c64e4.js" />
<script src="https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/js/hubdesignsystem.9c476c64e4.js" ></script>

        
    

<link rel="preload" as="script" href="https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/js/brands.34439a5a65.js" />
<script src="https://storage.googleapis.com/pe-portal-consumer-prod-wagtail-static/js/brands.34439a5a65.js" ></script>


    </body>

</html>
