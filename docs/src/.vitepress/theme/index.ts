// .vitepress/theme/index.ts
import { h } from "vue";
import DefaultTheme from "vitepress/theme";
import type { Theme as ThemeConfig } from "vitepress";

import {
  NolebaseEnhancedReadabilitiesMenu,
  NolebaseEnhancedReadabilitiesScreenMenu,
} from "@nolebase/vitepress-plugin-enhanced-readabilities/client";

import AsideTrustees from "../../components/AsideTrustees.vue";
import VersionPicker from "../../components/VersionPicker.vue";
import StarUs from "../../components/StarUs.vue";
import AuthorBadge from "../../components/AuthorBadge.vue";
import Authors from "../../components/Authors.vue";
import { enhanceAppWithTabs } from "vitepress-plugin-tabs/client";

import "@nolebase/vitepress-plugin-enhanced-readabilities/client/style.css";
import "./style.css";

export const Theme: ThemeConfig = {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      "aside-ads-before": () => h(AsideTrustees),
      "nav-bar-content-after": () => [
        h(StarUs),
        h(NolebaseEnhancedReadabilitiesMenu), // Enhanced Readabilities menu
      ],
      // A enhanced readabilities menu for narrower screens (usually smaller than iPad Mini)
      "nav-screen-content-after": () =>
        h(NolebaseEnhancedReadabilitiesScreenMenu),
    });
  },
  enhanceApp({ app, router, siteData }) {
    enhanceAppWithTabs(app);
    app.component("VersionPicker", VersionPicker);
    app.component("AuthorBadge", AuthorBadge);
    app.component("Authors", Authors);
  },
};
export default Theme;
