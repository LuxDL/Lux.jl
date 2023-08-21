<!-- <template>
    <div class="tutorial-demo-card">
        <div class="tutorial-demo-card-header">
            <h2>{{ title }}</h2>
        </div>
        <div class="tutorial-demo-card-content">
            <p>{{ description }}</p>
            <div class="tutorial-demo-card-actions">
                <a :href="demoLink" target="_blank" class="demo-link">View Demo</a>
                <a :href="tutorialLink" target="_blank" class="tutorial-link">Tutorial</a>
            </div>
        </div>
    </div>
</template>
  
<script>
export default {
    props: {
        title: String,
        description: String,
        demoLink: String,
        tutorialLink: String,
    },
};
</script>
  
<style scoped>
.tutorial-demo-card {
    border: 1px solid #ccc;
    border-radius: 4px;
    padding: 16px;
    margin: 16px 0;
    box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
}

.tutorial-demo-card-header {
    font-size: 1.5rem;
    margin-bottom: 8px;
}

.tutorial-demo-card-content {
    font-size: 1rem;
    margin-bottom: 8px;
}

.tutorial-demo-card-actions {
    display: flex;
    justify-content: space-between;
}

.demo-link,
.tutorial-link {
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    text-decoration: none;
    color: #333;
    transition: background-color 0.3s, color 0.3s;
}

.demo-link:hover,
.tutorial-link:hover {
    background-color: #f0f0f0;
    color: #000;
}
</style>
   -->

<script setup>
import { ref } from 'vue'

const count = ref(0)
</script>

<template>
    <button @click="count++">You clicked me {{ count }} times.</button>
</template>