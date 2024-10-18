import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.CxJX8IYj.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the spirals</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [MLUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Datasets</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">make_spiral</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sequence_length) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> _ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dataset_size]</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the labels</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    labels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">repeat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">([</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    clockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequence_length], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                         for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    anticlockwise_spirals </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">reshape</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                                 d[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">][:, (sequence_length </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], :, sequence_length, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">                             for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> d </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> data[((dataset_size </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">÷</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Float32</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(clockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, anticlockwise_spirals</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">...</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; dims</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Split the dataset</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (x_train, y_train), (x_val, y_val) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> splitobs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x_data, labels); at</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create DataLoaders</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Use DataLoader to automatically minibatch and shuffle the data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_train, y_train)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">),</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Don&#39;t shuffle the validation data</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        DataLoader</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">collect</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">.((x_val, y_val)); batchsize</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">128</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, shuffle</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">false</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR938/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR938/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR938/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # First we will have to run the sequence through the LSTM Cell</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # The first call to LSTM Cell will create the initial hidden state</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # See that the parameters and states are automatically populated into a field called</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`lstm_cell\` We use \`eachslice\` to get the elements in the sequence without copying,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # and \`Iterators.peel\` to split out the first element for LSTM initialization.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxOps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Now that we have the hidden state and memory in \`carry\` we will pass the input and</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`carry\` jointly</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        (y, carry), st_lstm </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry), ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">lstm_cell, st_lstm)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # After running through the sequence we will pass the output through the classifier</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y, st_classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">classifier)</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Finally remember to create the updated state</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> merge</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(st, (classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_classifier, lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">st_lstm))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y), st</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR938/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxOps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x_init)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x_rest</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y, carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((x, carry))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        @return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> vec</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">const</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> lossfn </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> BinaryCrossEntropyLoss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss, st_, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ŷ)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sum</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">((y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.5f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.==</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true)</span></span>
<span class="line"><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;">accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> matches</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">/</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> length</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>accuracy (generic function with 1 method)</span></span></code></pre></div><h2 id="Training-the-Model" tabindex="-1">Training the Model <a class="header-anchor" href="#Training-the-Model" aria-label="Permalink to &quot;Training the Model {#Training-the-Model}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model_type)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_loader, val_loader </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">() </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    ps, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">setup</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), lossfn, (x, y), train_state)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            ŷ, st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> lossfn</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ŷ, y)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation: Loss %4.5f Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62529</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59393</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56186</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54187</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52298</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49144</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47137</span></span>
<span class="line"><span>Validation: Loss 0.47690 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46454 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46300</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46489</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43176</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42881</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40319</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39728</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38930</span></span>
<span class="line"><span>Validation: Loss 0.38186 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36646 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36763</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35381</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35102</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31989</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32072</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31383</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31427</span></span>
<span class="line"><span>Validation: Loss 0.29792 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28097 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28122</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26590</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26624</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25853</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23910</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24324</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21958</span></span>
<span class="line"><span>Validation: Loss 0.22778 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21089 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20657</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20166</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20763</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19490</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18196</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16968</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17316</span></span>
<span class="line"><span>Validation: Loss 0.17096 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15557 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15191</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14507</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15185</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13586</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13964</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12911</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14062</span></span>
<span class="line"><span>Validation: Loss 0.12604 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11345 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11667</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10919</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10801</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09881</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10246</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08754</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09189</span></span>
<span class="line"><span>Validation: Loss 0.08999 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08080 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08541</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08028</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07488</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07366</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06574</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06082</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06145</span></span>
<span class="line"><span>Validation: Loss 0.06231 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05629 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05926</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05500</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05124</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04944</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05019</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04596</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04216</span></span>
<span class="line"><span>Validation: Loss 0.04628 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04208 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04341</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04263</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03996</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03654</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03950</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03721</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03497</span></span>
<span class="line"><span>Validation: Loss 0.03757 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03415 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03526</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03773</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03352</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03110</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03222</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02916</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02545</span></span>
<span class="line"><span>Validation: Loss 0.03197 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02900 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02891</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02957</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02898</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02819</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02643</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02709</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02753</span></span>
<span class="line"><span>Validation: Loss 0.02792 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02527 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02586</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02563</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02628</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02430</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02409</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02285</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02187</span></span>
<span class="line"><span>Validation: Loss 0.02475 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02236 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02291</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02313</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02058</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02202</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02205</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02134</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02102</span></span>
<span class="line"><span>Validation: Loss 0.02221 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02002 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01957</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02038</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01993</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02017</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01844</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01961</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02107</span></span>
<span class="line"><span>Validation: Loss 0.02009 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01807 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01714</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01725</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01955</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01877</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01804</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01679</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01659</span></span>
<span class="line"><span>Validation: Loss 0.01829 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01641 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01716</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01631</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01656</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01587</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01629</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01593</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01452</span></span>
<span class="line"><span>Validation: Loss 0.01674 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01500 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01554</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01594</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01547</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01441</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01379</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01521</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01163</span></span>
<span class="line"><span>Validation: Loss 0.01542 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01379 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01359</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01303</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01508</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01327</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01395</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01353</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01392</span></span>
<span class="line"><span>Validation: Loss 0.01428 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01275 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01336</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01296</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01166</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01395</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01304</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01141</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01290</span></span>
<span class="line"><span>Validation: Loss 0.01324 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01181 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01165</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01165</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01307</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01130</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01187</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01104</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01289</span></span>
<span class="line"><span>Validation: Loss 0.01227 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01094 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01149</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01201</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01188</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01055</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00967</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00953</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01271</span></span>
<span class="line"><span>Validation: Loss 0.01126 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01005 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01033</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01062</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00926</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01001</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01045</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00936</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00919</span></span>
<span class="line"><span>Validation: Loss 0.01013 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00906 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00980</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00816</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00843</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00846</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00888</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00979</span></span>
<span class="line"><span>Validation: Loss 0.00897 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00806 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00843</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00875</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00758</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00721</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00786</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00735</span></span>
<span class="line"><span>Validation: Loss 0.00806 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00727 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63625</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58885</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56031</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54287</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51723</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50624</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47496</span></span>
<span class="line"><span>Validation: Loss 0.46269 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46929 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46917</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45380</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44090</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43023</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41833</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39975</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36000</span></span>
<span class="line"><span>Validation: Loss 0.36451 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37210 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37049</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36640</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34469</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33487</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33437</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29816</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29173</span></span>
<span class="line"><span>Validation: Loss 0.27911 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28749 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29646</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27177</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26003</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25521</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24206</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24265</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24160</span></span>
<span class="line"><span>Validation: Loss 0.20931 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21774 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22515</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20107</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20535</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20395</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17171</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15734</span></span>
<span class="line"><span>Validation: Loss 0.15425 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16194 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16282</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15850</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14559</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14159</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14256</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12386</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13692</span></span>
<span class="line"><span>Validation: Loss 0.11245 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11878 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11171</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11641</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11495</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10014</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10080</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09365</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09590</span></span>
<span class="line"><span>Validation: Loss 0.08023 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08487 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08638</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08328</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07779</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07766</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06561</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06408</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05502</span></span>
<span class="line"><span>Validation: Loss 0.05586 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05892 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05622</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05991</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05255</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05021</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05018</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04824</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04682</span></span>
<span class="line"><span>Validation: Loss 0.04161 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04378 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04262</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04536</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04253</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04221</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03529</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03577</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03871</span></span>
<span class="line"><span>Validation: Loss 0.03364 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03540 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03768</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03757</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02971</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03336</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03421</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02924</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02965</span></span>
<span class="line"><span>Validation: Loss 0.02850 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03002 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03127</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02814</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02951</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02988</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02739</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02618</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02750</span></span>
<span class="line"><span>Validation: Loss 0.02477 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02613 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02859</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02296</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02561</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02557</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02429</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02451</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02189</span></span>
<span class="line"><span>Validation: Loss 0.02189 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02311 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02188</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02392</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02216</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02200</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02320</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02152</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01949</span></span>
<span class="line"><span>Validation: Loss 0.01958 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02070 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02088</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02002</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02100</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02033</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01985</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01850</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01961</span></span>
<span class="line"><span>Validation: Loss 0.01766 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01870 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01903</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01926</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01851</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01710</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01702</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01826</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01783</span></span>
<span class="line"><span>Validation: Loss 0.01604 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01700 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01619</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01691</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01577</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01758</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01521</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01699</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01935</span></span>
<span class="line"><span>Validation: Loss 0.01465 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01554 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01507</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01489</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01546</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01541</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01622</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01447</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01333</span></span>
<span class="line"><span>Validation: Loss 0.01345 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01428 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01538</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01548</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01311</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01419</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01335</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01203</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01540</span></span>
<span class="line"><span>Validation: Loss 0.01241 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01318 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01278</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01225</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01415</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01400</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01187</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01284</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01134</span></span>
<span class="line"><span>Validation: Loss 0.01148 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01220 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01219</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01278</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01228</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01168</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01189</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01109</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01141</span></span>
<span class="line"><span>Validation: Loss 0.01062 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01128 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01205</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01003</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01090</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01141</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01116</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01091</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00966</span></span>
<span class="line"><span>Validation: Loss 0.00973 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01031 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00908</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01079</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01093</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00916</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01001</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00868</span></span>
<span class="line"><span>Validation: Loss 0.00873 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00924 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00986</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00872</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00907</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00885</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00901</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00827</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00773</span></span>
<span class="line"><span>Validation: Loss 0.00777 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00821 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00865</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00830</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00810</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00741</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00808</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00674</span></span>
<span class="line"><span>Validation: Loss 0.00705 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00744 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(MLDataDevices)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> MLDataDevices</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.5</span></span>
<span class="line"><span>Commit 6f3fdf7b362 (2024-08-27 14:19 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 48 default, 0 interactive, 24 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 48</span></span>
<span class="line"><span>  JULIA_CUDA_HARD_MEMORY_LIMIT = 100%</span></span>
<span class="line"><span>  JULIA_PKG_PRECOMPILE_AUTO = 0</span></span>
<span class="line"><span>  JULIA_DEBUG = Literate</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA runtime 12.5, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.5</span></span>
<span class="line"><span>NVIDIA driver 555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.5.3</span></span>
<span class="line"><span>- CURAND: 10.3.6</span></span>
<span class="line"><span>- CUFFT: 11.2.3</span></span>
<span class="line"><span>- CUSOLVER: 11.6.3</span></span>
<span class="line"><span>- CUSPARSE: 12.5.1</span></span>
<span class="line"><span>- CUPTI: 2024.2.1 (API 23.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.4.3</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.9.2+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.14.1+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.5</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Preferences:</span></span>
<span class="line"><span>- CUDA_Driver_jll.compat: false</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
