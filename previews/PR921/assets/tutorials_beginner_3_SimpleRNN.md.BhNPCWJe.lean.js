import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.C_ZzdWYP.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR921/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR921/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR921/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR921/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.60728</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.60144</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54850</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53119</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53768</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50759</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47549</span></span>
<span class="line"><span>Validation: Loss 0.47933 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47764 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46961</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45510</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44510</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43714</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40112</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39816</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37707</span></span>
<span class="line"><span>Validation: Loss 0.38546 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.38378 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37145</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35654</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35389</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32933</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31838</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32197</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31645</span></span>
<span class="line"><span>Validation: Loss 0.30257 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.30078 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29676</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28598</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26980</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25274</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24825</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23357</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21381</span></span>
<span class="line"><span>Validation: Loss 0.23265 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.23083 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20471</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20188</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21146</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20601</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18449</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18742</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16740</span></span>
<span class="line"><span>Validation: Loss 0.17592 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17405 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16611</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15699</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14916</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15036</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13154</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13379</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12866</span></span>
<span class="line"><span>Validation: Loss 0.13039 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12868 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12280</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10664</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11601</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10791</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09881</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09362</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09915</span></span>
<span class="line"><span>Validation: Loss 0.09348 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09215 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.09592</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08207</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07828</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06886</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06712</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06562</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06622</span></span>
<span class="line"><span>Validation: Loss 0.06457 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06366 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05857</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05221</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06156</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05183</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04753</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04855</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04850</span></span>
<span class="line"><span>Validation: Loss 0.04780 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04718 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04706</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04058</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03927</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03978</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04038</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03788</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04270</span></span>
<span class="line"><span>Validation: Loss 0.03868 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03818 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03524</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03494</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03318</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03299</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03424</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03120</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03610</span></span>
<span class="line"><span>Validation: Loss 0.03284 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03240 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03256</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02866</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02914</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02828</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02617</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02918</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02645</span></span>
<span class="line"><span>Validation: Loss 0.02855 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02816 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02589</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02710</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02591</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02431</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02521</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02389</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02240</span></span>
<span class="line"><span>Validation: Loss 0.02526 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02491 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02527</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02293</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02298</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02302</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01979</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02112</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02078</span></span>
<span class="line"><span>Validation: Loss 0.02263 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02231 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02165</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02113</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02042</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02071</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01946</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01787</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01873</span></span>
<span class="line"><span>Validation: Loss 0.02045 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02016 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01826</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01999</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01865</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01646</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01886</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01757</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01629</span></span>
<span class="line"><span>Validation: Loss 0.01861 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01834 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01930</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01736</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01664</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01692</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01475</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01552</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01297</span></span>
<span class="line"><span>Validation: Loss 0.01702 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01676 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01639</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01618</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01722</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01365</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01363</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01436</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01358</span></span>
<span class="line"><span>Validation: Loss 0.01562 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01538 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01519</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01439</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01252</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01334</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01333</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01402</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01630</span></span>
<span class="line"><span>Validation: Loss 0.01432 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01410 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01364</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01334</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01275</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01259</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01164</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01273</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01080</span></span>
<span class="line"><span>Validation: Loss 0.01296 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01277 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01188</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01333</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01231</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01047</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01099</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01047</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00795</span></span>
<span class="line"><span>Validation: Loss 0.01147 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01130 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01138</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01045</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00903</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00963</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00852</span></span>
<span class="line"><span>Validation: Loss 0.01012 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00997 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00975</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00897</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00886</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00940</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00856</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00901</span></span>
<span class="line"><span>Validation: Loss 0.00914 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00900 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00795</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00784</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00834</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00810</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00888</span></span>
<span class="line"><span>Validation: Loss 0.00842 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00830 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00784</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00733</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00811</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00712</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00755</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00775</span></span>
<span class="line"><span>Validation: Loss 0.00787 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00776 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63463</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57975</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56646</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53690</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52765</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49979</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48828</span></span>
<span class="line"><span>Validation: Loss 0.46332 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47726 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46765</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45935</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44733</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42212</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41083</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40231</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37149</span></span>
<span class="line"><span>Validation: Loss 0.36532 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.38131 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37351</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35460</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34964</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33138</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32155</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31004</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32745</span></span>
<span class="line"><span>Validation: Loss 0.27969 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29730 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28998</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26675</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28140</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24914</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24666</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23107</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24976</span></span>
<span class="line"><span>Validation: Loss 0.20978 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22731 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21764</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20450</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20979</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18447</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18162</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18606</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15935</span></span>
<span class="line"><span>Validation: Loss 0.15449 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17065 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16715</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15300</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14692</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14689</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12963</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13313</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12042</span></span>
<span class="line"><span>Validation: Loss 0.11264 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12611 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12609</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11834</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11714</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09605</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08784</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09400</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09284</span></span>
<span class="line"><span>Validation: Loss 0.08056 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09058 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08425</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08032</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07774</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07493</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06512</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06816</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06959</span></span>
<span class="line"><span>Validation: Loss 0.05626 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06297 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06238</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05505</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05750</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04623</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04970</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04620</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05311</span></span>
<span class="line"><span>Validation: Loss 0.04178 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04648 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04283</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04276</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03971</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03784</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04150</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03756</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04253</span></span>
<span class="line"><span>Validation: Loss 0.03376 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03760 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03371</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03434</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03460</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03379</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03167</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03225</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03288</span></span>
<span class="line"><span>Validation: Loss 0.02859 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03192 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03063</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02744</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02984</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03121</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02681</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02626</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02667</span></span>
<span class="line"><span>Validation: Loss 0.02483 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02779 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02528</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02593</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02570</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02370</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02674</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02330</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02357</span></span>
<span class="line"><span>Validation: Loss 0.02194 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02461 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02462</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02419</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02241</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02067</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02016</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02195</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02105</span></span>
<span class="line"><span>Validation: Loss 0.01961 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02206 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02147</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02042</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01885</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02073</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02007</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01892</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01828</span></span>
<span class="line"><span>Validation: Loss 0.01769 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01995 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01889</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01864</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01837</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01864</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01657</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01799</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01661</span></span>
<span class="line"><span>Validation: Loss 0.01607 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01817 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01720</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01665</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01644</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01686</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01634</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01614</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01457</span></span>
<span class="line"><span>Validation: Loss 0.01469 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01666 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01484</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01570</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01508</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01474</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01603</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01427</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01624</span></span>
<span class="line"><span>Validation: Loss 0.01352 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01536 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01309</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01447</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01321</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01416</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01591</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01331</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01291</span></span>
<span class="line"><span>Validation: Loss 0.01250 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01422 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01250</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01445</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01249</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01389</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01209</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01235</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01298</span></span>
<span class="line"><span>Validation: Loss 0.01160 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01321 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01345</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01147</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01221</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01197</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01218</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01148</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01019</span></span>
<span class="line"><span>Validation: Loss 0.01078 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01228 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01089</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01078</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01192</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01124</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01143</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01110</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01034</span></span>
<span class="line"><span>Validation: Loss 0.01000 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01139 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01108</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01084</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01039</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01007</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01023</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01001</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00832</span></span>
<span class="line"><span>Validation: Loss 0.00916 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01041 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01013</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00971</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00911</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01013</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00863</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00902</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00819</span></span>
<span class="line"><span>Validation: Loss 0.00822 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00930 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00873</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00891</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00830</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00815</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00772</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00704</span></span>
<span class="line"><span>Validation: Loss 0.00734 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00827 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
