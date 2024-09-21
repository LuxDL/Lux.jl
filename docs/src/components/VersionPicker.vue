<!-- Adapted from https://github.com/MakieOrg/Makie.jl/blob/master/docs/src/.vitepress/theme/VersionPicker.vue -->

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useData } from 'vitepress'
import VPNavBarMenuGroup from 'vitepress/dist/client/theme-default/components/VPNavBarMenuGroup.vue'
import VPNavScreenMenuGroup from 'vitepress/dist/client/theme-default/components/VPNavScreenMenuGroup.vue'

// Extend the global Window interface to include DOC_VERSIONS and DOCUMENTER_CURRENT_VERSION
declare global {
  interface Window {
    DOC_VERSIONS?: string[];
    DOCUMENTER_CURRENT_VERSION?: string;
  }
}

const props = defineProps<{
  screenMenu?: boolean
}>()

const versions = ref<Array<{ text: string, link: string }>>([]);
const currentVersion = ref('Versions');
const isClient = ref(false);
const { site } = useData()

const isLocalBuild = () => {
  return typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1');
}

const getBaseRepository = () => {
  if (typeof window === 'undefined') return ''; // Handle server-side rendering (SSR)
  const { origin, pathname } = window.location;
  // Check if it's a GitHub Pages (or similar) setup
  if (origin.includes('github.io')) {
    // Extract the first part of the path as the repository name
    const pathParts = pathname.split('/').filter(Boolean);
    const baseRepo = pathParts.length > 0 ? `/${pathParts[0]}/` : '/';
    return `${origin}${baseRepo}`;
  } else {
    // For custom domains, use just the origin (e.g., https://docs.makie.org)
    return origin;
  }
};

const waitForScriptsToLoad = () => {
  return new Promise<boolean>((resolve) => {
    if (isLocalBuild()) {
      resolve(false);
      return;
    }
    const checkInterval = setInterval(() => {
      if (window.DOC_VERSIONS && window.DOCUMENTER_CURRENT_VERSION) {
        clearInterval(checkInterval);
        resolve(true);
      }
    }, 100);
    // Timeout after 5 seconds
    setTimeout(() => {
      clearInterval(checkInterval);
      resolve(false);
    }, 5000);
  });
};

const loadVersions = async () => {
  if (typeof window === 'undefined') return; // Guard for SSR

  try {
    if (isLocalBuild()) {
      // Handle the local build scenario directly
      const fallbackVersions = ['dev'];
      versions.value = fallbackVersions.map(v => ({
        text: v,
        link: '/'
      }));
      currentVersion.value = 'dev';
    } else {
      // For non-local builds, wait for scripts to load
      const scriptsLoaded = await waitForScriptsToLoad();
      const getBaseRepositoryPath = computed(() => {
        return getBaseRepository();
      });

      if (scriptsLoaded && window.DOC_VERSIONS && window.DOCUMENTER_CURRENT_VERSION) {
        versions.value = window.DOC_VERSIONS.map((v: string) => ({
          text: v,
          link: `${getBaseRepositoryPath.value}/${v}/`
        }));
        currentVersion.value = window.DOCUMENTER_CURRENT_VERSION;
      } else {
        // Fallback logic if scripts fail to load or are not available
        const fallbackVersions = ['dev'];
        versions.value = fallbackVersions.map(v => ({
          text: v,
          link: `${getBaseRepositoryPath.value}/${v}/`
        }));
        currentVersion.value = 'dev';
      }
    }
  } catch (error) {
    console.warn('Error loading versions:', error);
    // Use fallback logic in case of an error
    const fallbackVersions = ['dev'];
    const getBaseRepositoryPath = computed(() => {
        return getBaseRepository();
      });
    versions.value = fallbackVersions.map(v => ({
      text: v,
      link: `${getBaseRepositoryPath.value}/${v}/`
    }));
    currentVersion.value = 'dev';
  }
  isClient.value = true;
};

onMounted(loadVersions);
</script>

<template>
  <template v-if="isClient">
    <VPNavBarMenuGroup
      v-if="!screenMenu && versions.length > 0"
      :item="{ text: currentVersion, items: versions }"
      class="VPVersionPicker"
    />
    <VPNavScreenMenuGroup
      v-else-if="screenMenu && versions.length > 0"
      :text="currentVersion"
      :items="versions"
      class="VPVersionPicker"
    />
  </template>
</template>

<style scoped>
.VPVersionPicker :deep(button .text) {
  color: var(--vp-c-text-1) !important;
}
.VPVersionPicker:hover :deep(button .text) {
  color: var(--vp-c-text-2) !important;
}
</style>
