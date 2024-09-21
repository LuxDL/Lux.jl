// .vitepress/theme/index.ts
import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import AsideTrustees from '../../components/AsideTrustees.vue'
import VersionPicker from "../../components/VersionPicker.vue";

import { enhanceAppWithTabs } from 'vitepress-plugin-tabs/client'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout() {
    return h(DefaultTheme.Layout, null, {
      'aside-ads-before': () => h(AsideTrustees),
    })
  },
  enhanceApp({ app }) {
    enhanceAppWithTabs(app);
    app.component('VersionPicker', VersionPicker);
  }
} satisfies Theme