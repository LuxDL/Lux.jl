import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.CvbbdR01.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR905/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR905/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR905/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR905/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61222</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59394</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56420</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54589</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52186</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49909</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49502</span></span>
<span class="line"><span>Validation: Loss 0.46906 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.45922 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47247</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45383</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43742</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42464</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41341</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40073</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38038</span></span>
<span class="line"><span>Validation: Loss 0.37224 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36043 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37854</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35935</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36085</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32962</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31188</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30124</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30276</span></span>
<span class="line"><span>Validation: Loss 0.28777 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27444 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28167</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27343</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26410</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25763</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24202</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24145</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25774</span></span>
<span class="line"><span>Validation: Loss 0.21832 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20462 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22883</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20457</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19942</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18484</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18627</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17741</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16772</span></span>
<span class="line"><span>Validation: Loss 0.16271 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15020 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16499</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14703</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14776</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13755</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13776</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13416</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14425</span></span>
<span class="line"><span>Validation: Loss 0.11970 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10932 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12155</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11104</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11214</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10222</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09316</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09861</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08895</span></span>
<span class="line"><span>Validation: Loss 0.08589 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07818 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08725</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07874</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08013</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07677</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07094</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06160</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05602</span></span>
<span class="line"><span>Validation: Loss 0.05979 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05465 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06276</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05443</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05676</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04999</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04974</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04547</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04390</span></span>
<span class="line"><span>Validation: Loss 0.04399 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04045 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04708</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04238</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04200</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03942</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03820</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03446</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03378</span></span>
<span class="line"><span>Validation: Loss 0.03542 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03257 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03316</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03451</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03312</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03283</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03336</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03237</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02943</span></span>
<span class="line"><span>Validation: Loss 0.03003 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02755 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02953</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02963</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02807</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02445</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02856</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02940</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02995</span></span>
<span class="line"><span>Validation: Loss 0.02612 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02390 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02500</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02627</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02604</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02449</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02220</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02425</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02729</span></span>
<span class="line"><span>Validation: Loss 0.02307 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02108 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02286</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02482</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02290</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02156</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02065</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01998</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01999</span></span>
<span class="line"><span>Validation: Loss 0.02062 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01880 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02026</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02030</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02106</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01917</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01855</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01947</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01873</span></span>
<span class="line"><span>Validation: Loss 0.01861 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01693 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02028</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01785</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01759</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01655</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01742</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01815</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01573</span></span>
<span class="line"><span>Validation: Loss 0.01691 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01535 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01660</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01730</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01738</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01565</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01591</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01555</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01374</span></span>
<span class="line"><span>Validation: Loss 0.01547 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01401 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01517</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01471</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01553</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01533</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01389</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01524</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01371</span></span>
<span class="line"><span>Validation: Loss 0.01424 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01287 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01274</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01433</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01351</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01361</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01452</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01423</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01215</span></span>
<span class="line"><span>Validation: Loss 0.01316 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01188 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01265</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01314</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01399</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01199</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01235</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01270</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01126</span></span>
<span class="line"><span>Validation: Loss 0.01220 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01100 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01254</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01254</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01228</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01238</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01030</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01130</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00997</span></span>
<span class="line"><span>Validation: Loss 0.01129 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01017 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01127</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01148</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01070</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01120</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01044</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01082</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00874</span></span>
<span class="line"><span>Validation: Loss 0.01035 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00933 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01025</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00911</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01015</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00936</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01039</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01013</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01009</span></span>
<span class="line"><span>Validation: Loss 0.00930 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00839 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00946</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00904</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00822</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00864</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00914</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00872</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00796</span></span>
<span class="line"><span>Validation: Loss 0.00824 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00747 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00740</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00768</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00749</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00783</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00615</span></span>
<span class="line"><span>Validation: Loss 0.00744 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00678 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62400</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59807</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57011</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53576</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51980</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50250</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48332</span></span>
<span class="line"><span>Validation: Loss 0.46471 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.45769 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47095</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45331</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44636</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42368</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41170</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39691</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38279</span></span>
<span class="line"><span>Validation: Loss 0.36616 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.35808 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37494</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35974</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34074</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33628</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32017</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31195</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.28080</span></span>
<span class="line"><span>Validation: Loss 0.28021 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27111 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29098</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26956</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26052</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25288</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24151</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24411</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22936</span></span>
<span class="line"><span>Validation: Loss 0.21029 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20076 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21416</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20588</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18593</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20320</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18878</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17736</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15669</span></span>
<span class="line"><span>Validation: Loss 0.15538 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.14636 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16292</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15943</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14700</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14210</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12674</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12869</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13081</span></span>
<span class="line"><span>Validation: Loss 0.11360 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10601 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11740</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10753</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11586</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10066</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09846</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09310</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08795</span></span>
<span class="line"><span>Validation: Loss 0.08133 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07564 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08573</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08302</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07143</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07792</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06464</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06788</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05493</span></span>
<span class="line"><span>Validation: Loss 0.05674 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05295 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05822</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05619</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05716</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05249</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04977</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04265</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04171</span></span>
<span class="line"><span>Validation: Loss 0.04189 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03929 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04123</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04242</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04071</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04004</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03793</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03738</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03669</span></span>
<span class="line"><span>Validation: Loss 0.03374 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03164 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03621</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03423</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03259</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03411</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03155</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02986</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02600</span></span>
<span class="line"><span>Validation: Loss 0.02855 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02673 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02785</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03101</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02829</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02793</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02854</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02491</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02765</span></span>
<span class="line"><span>Validation: Loss 0.02481 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02319 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02759</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02435</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02328</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02429</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02467</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02367</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02291</span></span>
<span class="line"><span>Validation: Loss 0.02191 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02044 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02268</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02269</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02251</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02049</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02215</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02112</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01902</span></span>
<span class="line"><span>Validation: Loss 0.01959 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01824 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01976</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02121</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02037</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01873</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01912</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01784</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02168</span></span>
<span class="line"><span>Validation: Loss 0.01766 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01642 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01813</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01966</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01795</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01803</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01656</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01600</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01791</span></span>
<span class="line"><span>Validation: Loss 0.01602 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01488 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01647</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01566</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01532</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01659</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01694</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01544</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01750</span></span>
<span class="line"><span>Validation: Loss 0.01464 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01356 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01534</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01502</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01386</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01490</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01515</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01430</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01504</span></span>
<span class="line"><span>Validation: Loss 0.01344 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01244 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01371</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01290</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01404</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01311</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01396</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01382</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01389</span></span>
<span class="line"><span>Validation: Loss 0.01241 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01148 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01290</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01408</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01202</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01225</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01188</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01266</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01151</span></span>
<span class="line"><span>Validation: Loss 0.01149 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01062 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01229</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01126</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01180</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01211</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01056</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01218</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01067</span></span>
<span class="line"><span>Validation: Loss 0.01065 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00984 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01107</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01114</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00969</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01068</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01078</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01072</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01296</span></span>
<span class="line"><span>Validation: Loss 0.00979 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00904 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01084</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01011</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00943</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01094</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00963</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00818</span></span>
<span class="line"><span>Validation: Loss 0.00882 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00816 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00897</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00869</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00908</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00880</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00841</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00847</span></span>
<span class="line"><span>Validation: Loss 0.00784 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00729 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00776</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00786</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00802</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00795</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00769</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00771</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00762</span></span>
<span class="line"><span>Validation: Loss 0.00708 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00660 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.484 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
