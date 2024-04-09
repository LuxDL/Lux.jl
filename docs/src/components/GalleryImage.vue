<script setup lang="ts">
import { withBase } from 'vitepress'
export interface Props {
    href: string;
    src: string;
    caption: string;
    desc: string;
}

defineProps<Props>();
</script>

<template>
    <div class="img-box">
        <a :href="href">
            <img :src="withBase(src)" height="150px" alt="">
            <div class="transparent-box1">
                <div class="caption">
                    <h2>{{ caption }}</h2>
                </div>
            </div>
            <div class="transparent-box2">
                <div class="subcaption">
                    <p class="opacity-low">{{ desc }}</p>
                </div>
            </div>
        </a>
    </div>
</template>

<style scoped>
.img-box {
    box-sizing: content-box;
    border-radius: 14px;
    margin: 20px;
    height: 350px;
    width: 250px;
    overflow: hidden;
    display: inline-block;
    color: white;
    position: relative;
    background-color: transparent;
    border: 2px solid var(--vp-c-bg-alt);
}

.img-box h2 {
    border-top: 0;
}

.img-box img {
    height: 100%;
    width: 100%;
    object-fit: cover;
    opacity: 0.3;
    transition: transform 0.3s ease, opacity 0.3s ease;
}

.caption {
    position: absolute;
    bottom: 30px;
    color: var(--vp-c-text-1);
    left: 10px;
    opacity: 1;
    transition: transform 0.3s ease, opacity 0.3s ease;
}

.subcaption {
    position: absolute;
    bottom: 5px;
    color: var(--vp-c-text-1);
    left: 10px;
    opacity: 0;
    transition: transform 0.3s ease, opacity 0.3s ease;
}

.transparent-box1 {
    height: 250px;
    width: 250px;
    background-color: transparent;
    position: absolute;
    top: 0;
    left: 0;
    transition: background-color 0.3s ease;
}

.transparent-box2 {
    height: 100px;
    width: 250px;
    background-color: transparent;
    position: absolute;
    top: 250px;
    left: 0;
    transition: background-color 0.3s ease;
}

.img-box:hover img {
    transform: scale(1.1);
}

.img-box:hover .transparent-box1 {
    background-color: var(--vp-c-bg-alt);
}

.img-box:hover .transparent-box2 {
    background-color: var(--vp-c-bg-alt);
}

.img-box:hover .caption {
    transform: translateY(-20px);
    opacity: 1;
}

.img-box:hover .subcaption {
    transform: translateY(-20px);
    opacity: 1;
}

.img-box:hover {
    border: 2px solid var(--vp-c-brand-light);
    cursor: pointer;
}

.caption>p:nth-child(2) {
    font-size: 0.8em;
}
.subcaption>p:nth-child(2) {
    font-size: 0.8em;
}

.opacity-low {
    opacity: 0.85;
}
</style>