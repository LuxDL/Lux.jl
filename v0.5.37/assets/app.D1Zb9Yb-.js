import { V as inBrowser, aa as useUpdateHead, ab as RouterSymbol, ac as initData, ad as dataSymbol, ae as Content, af as ClientOnly, ag as siteDataRef, ah as createSSRApp, ai as createRouter, aj as pathToFile, Y as __vitePreload, d as defineComponent, u as useData, k as onMounted, y as watchEffect, ak as usePrefetch, al as useCopyCode, am as useCodeGroups, a7 as h } from "./chunks/framework.C_ldBCUa.js";
import { R as RawTheme } from "./chunks/theme.DuDy91Kj.js";
function resolveThemeExtends(theme) {
  if (theme.extends) {
    const base = resolveThemeExtends(theme.extends);
    return {
      ...base,
      ...theme,
      async enhanceApp(ctx) {
        if (base.enhanceApp)
          await base.enhanceApp(ctx);
        if (theme.enhanceApp)
          await theme.enhanceApp(ctx);
      }
    };
  }
  return theme;
}
const Theme = resolveThemeExtends(RawTheme);
const VitePressApp = defineComponent({
  name: "VitePressApp",
  setup() {
    const { site, lang, dir } = useData();
    onMounted(() => {
      watchEffect(() => {
        document.documentElement.lang = lang.value;
        document.documentElement.dir = dir.value;
      });
    });
    if (site.value.router.prefetchLinks) {
      usePrefetch();
    }
    useCopyCode();
    useCodeGroups();
    if (Theme.setup)
      Theme.setup();
    return () => h(Theme.Layout);
  }
});
async function createApp() {
  globalThis.__VITEPRESS__ = true;
  const router = newRouter();
  const app = newApp();
  app.provide(RouterSymbol, router);
  const data = initData(router.route);
  app.provide(dataSymbol, data);
  app.component("Content", Content);
  app.component("ClientOnly", ClientOnly);
  Object.defineProperties(app.config.globalProperties, {
    $frontmatter: {
      get() {
        return data.frontmatter.value;
      }
    },
    $params: {
      get() {
        return data.page.value.params;
      }
    }
  });
  if (Theme.enhanceApp) {
    await Theme.enhanceApp({
      app,
      router,
      siteData: siteDataRef
    });
  }
  return { app, router, data };
}
function newApp() {
  return createSSRApp(VitePressApp);
}
function newRouter() {
  let isInitialPageLoad = inBrowser;
  let initialPath;
  return createRouter((path) => {
    let pageFilePath = pathToFile(path);
    let pageModule = null;
    if (pageFilePath) {
      if (isInitialPageLoad) {
        initialPath = pageFilePath;
      }
      if (isInitialPageLoad || initialPath === pageFilePath) {
        pageFilePath = pageFilePath.replace(/\.js$/, ".lean.js");
      }
      if (false)
        ;
      else {
        pageModule = __vitePreload(() => import(
          /*@vite-ignore*/
          pageFilePath
        ), true ? [] : void 0);
      }
    }
    if (inBrowser) {
      isInitialPageLoad = false;
    }
    return pageModule;
  }, Theme.NotFound);
}
if (inBrowser) {
  createApp().then(({ app, router, data }) => {
    router.go().then(() => {
      useUpdateHead(router.route, data.site);
      app.mount("#app");
    });
  });
}
export {
  createApp
};
