import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.BsSavNHR.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR1099/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR1099/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR1099/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR1099/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62792</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60848</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55934</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54185</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51294</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50765</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48027</span></span>
<span class="line"><span>Validation: Loss 0.46639 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.45089 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47696</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47125</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43931</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41957</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40204</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39903</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39892</span></span>
<span class="line"><span>Validation: Loss 0.36880 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.35080 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37989</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35641</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34153</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34154</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31592</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31488</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31848</span></span>
<span class="line"><span>Validation: Loss 0.28344 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26339 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28470</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27408</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26613</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25058</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26197</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24043</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22592</span></span>
<span class="line"><span>Validation: Loss 0.21356 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.19310 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21826</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21196</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20437</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19399</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18408</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17805</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16271</span></span>
<span class="line"><span>Validation: Loss 0.15824 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.13902 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16333</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15907</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14982</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13759</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13508</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13489</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13243</span></span>
<span class="line"><span>Validation: Loss 0.11578 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09958 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11267</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11373</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11169</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10519</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10695</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09479</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07887</span></span>
<span class="line"><span>Validation: Loss 0.08275 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07066 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08068</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08179</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08182</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06899</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07043</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06918</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06629</span></span>
<span class="line"><span>Validation: Loss 0.05776 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04967 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05783</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05452</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05804</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05255</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05152</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04826</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03783</span></span>
<span class="line"><span>Validation: Loss 0.04307 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03733 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04416</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04233</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04099</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04310</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04021</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03595</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03778</span></span>
<span class="line"><span>Validation: Loss 0.03493 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03022 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03465</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03613</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03552</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03185</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03339</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03181</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03407</span></span>
<span class="line"><span>Validation: Loss 0.02967 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02558 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03202</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02725</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02924</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02839</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02855</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02996</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02583</span></span>
<span class="line"><span>Validation: Loss 0.02583 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02219 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02561</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02718</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02447</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02625</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02601</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02428</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02281</span></span>
<span class="line"><span>Validation: Loss 0.02286 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01958 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02245</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02475</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02131</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02278</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02362</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02171</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02118</span></span>
<span class="line"><span>Validation: Loss 0.02048 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01749 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02143</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02113</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02081</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02031</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02129</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01761</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02053</span></span>
<span class="line"><span>Validation: Loss 0.01850 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01575 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01816</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01946</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01833</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01926</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01802</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01943</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01214</span></span>
<span class="line"><span>Validation: Loss 0.01682 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01428 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01666</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01933</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01672</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01745</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01629</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01515</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01589</span></span>
<span class="line"><span>Validation: Loss 0.01539 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01301 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01583</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01715</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01576</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01600</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01566</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01334</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01229</span></span>
<span class="line"><span>Validation: Loss 0.01413 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01191 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01455</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01523</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01395</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01461</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01348</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01334</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01558</span></span>
<span class="line"><span>Validation: Loss 0.01302 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01095 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01276</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01301</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01274</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01400</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01269</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01306</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01539</span></span>
<span class="line"><span>Validation: Loss 0.01200 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01008 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01214</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01273</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01196</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01276</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01038</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01266</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01183</span></span>
<span class="line"><span>Validation: Loss 0.01098 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00923 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01123</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01105</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01069</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01127</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01057</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01090</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01185</span></span>
<span class="line"><span>Validation: Loss 0.00985 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00832 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00987</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01035</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00935</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01008</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00903</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00903</span></span>
<span class="line"><span>Validation: Loss 0.00870 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00740 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00864</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00858</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00957</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00843</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00856</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00842</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00777</span></span>
<span class="line"><span>Validation: Loss 0.00782 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00668 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00739</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00801</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00815</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00751</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00783</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00741</span></span>
<span class="line"><span>Validation: Loss 0.00719 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00617 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61998</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59132</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55927</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54553</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51767</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49937</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47973</span></span>
<span class="line"><span>Validation: Loss 0.47145 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47557 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47750</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43977</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43440</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42685</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41246</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40053</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38500</span></span>
<span class="line"><span>Validation: Loss 0.37519 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.38009 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37372</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35428</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34618</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32708</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32369</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30772</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29411</span></span>
<span class="line"><span>Validation: Loss 0.29051 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29576 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29323</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27245</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25711</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25408</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24890</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22532</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22980</span></span>
<span class="line"><span>Validation: Loss 0.22009 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22529 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21686</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21663</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19319</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18539</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17371</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16957</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18713</span></span>
<span class="line"><span>Validation: Loss 0.16345 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16822 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15898</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15155</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14869</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13528</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13368</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12575</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11818</span></span>
<span class="line"><span>Validation: Loss 0.11954 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12344 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11039</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11049</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10727</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10035</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10007</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08965</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08444</span></span>
<span class="line"><span>Validation: Loss 0.08541 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08830 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08113</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07895</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07846</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06752</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06886</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06041</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06523</span></span>
<span class="line"><span>Validation: Loss 0.05936 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06130 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05194</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05608</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05565</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04953</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04864</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04490</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04616</span></span>
<span class="line"><span>Validation: Loss 0.04411 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04548 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04443</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04195</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03718</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03669</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03565</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03927</span></span>
<span class="line"><span>Validation: Loss 0.03572 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03685 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03515</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03592</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03207</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03027</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03087</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03046</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03221</span></span>
<span class="line"><span>Validation: Loss 0.03033 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03131 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03082</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02970</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02955</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02365</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02774</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02606</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02534</span></span>
<span class="line"><span>Validation: Loss 0.02639 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02726 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02487</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02549</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02399</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02314</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02418</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02436</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02394</span></span>
<span class="line"><span>Validation: Loss 0.02336 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02415 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02414</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02104</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02164</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02320</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02135</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01917</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01851</span></span>
<span class="line"><span>Validation: Loss 0.02091 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02163 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01877</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01953</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01872</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02001</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02024</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01876</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02069</span></span>
<span class="line"><span>Validation: Loss 0.01890 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01957 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01739</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01864</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01806</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01749</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01741</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01702</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01505</span></span>
<span class="line"><span>Validation: Loss 0.01718 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01781 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01749</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01712</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01629</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01532</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01543</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01436</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01609</span></span>
<span class="line"><span>Validation: Loss 0.01573 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01631 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01530</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01268</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01579</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01518</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01500</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01442</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01317</span></span>
<span class="line"><span>Validation: Loss 0.01449 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01504 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01327</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01313</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01405</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01507</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01283</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01333</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01165</span></span>
<span class="line"><span>Validation: Loss 0.01342 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01393 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01241</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01353</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01338</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01288</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01153</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01189</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01130</span></span>
<span class="line"><span>Validation: Loss 0.01247 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01295 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01176</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01129</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01209</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01140</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01197</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01162</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01095</span></span>
<span class="line"><span>Validation: Loss 0.01159 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01204 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01082</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01223</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01038</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01155</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00942</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01099</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00931</span></span>
<span class="line"><span>Validation: Loss 0.01073 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01115 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01053</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01050</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00998</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00996</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01012</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00932</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00817</span></span>
<span class="line"><span>Validation: Loss 0.00979 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01017 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01006</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00803</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00845</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00961</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00862</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00946</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00900</span></span>
<span class="line"><span>Validation: Loss 0.00875 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00908 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00827</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00906</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00844</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00803</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00710</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00763</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00773</span></span>
<span class="line"><span>Validation: Loss 0.00780 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00808 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.6</span></span>
<span class="line"><span>Commit 67dffc4a8ae (2024-10-28 12:23 UTC)</span></span>
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
<span class="line"><span>CUDA runtime 12.6, artifact installation</span></span>
<span class="line"><span>CUDA driver 12.6</span></span>
<span class="line"><span>NVIDIA driver 560.35.3</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.6.3</span></span>
<span class="line"><span>- CURAND: 10.3.7</span></span>
<span class="line"><span>- CUFFT: 11.3.0</span></span>
<span class="line"><span>- CUSOLVER: 11.7.1</span></span>
<span class="line"><span>- CUSPARSE: 12.5.4</span></span>
<span class="line"><span>- CUPTI: 2024.3.2 (API 24.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+560.35.3</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.5.2</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.10.3+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.15.4+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.6</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.422 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
