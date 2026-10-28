import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.DzwdoL1v.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.2.3/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.2.3/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.2.3/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.2.3/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62225</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58452</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57647</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53713</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51691</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49888</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48847</span></span>
<span class="line"><span>Validation: Loss 0.47424 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46706 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47281</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45886</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44245</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42243</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40542</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39367</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38477</span></span>
<span class="line"><span>Validation: Loss 0.37763 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36948 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36578</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36006</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35264</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33993</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30781</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30599</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30994</span></span>
<span class="line"><span>Validation: Loss 0.29307 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28429 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29298</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27158</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26401</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25578</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23678</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23406</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23145</span></span>
<span class="line"><span>Validation: Loss 0.22292 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21429 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.22214</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20231</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18991</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18320</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19426</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17266</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17777</span></span>
<span class="line"><span>Validation: Loss 0.16650 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15865 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15678</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15592</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14733</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13975</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13619</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12549</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12288</span></span>
<span class="line"><span>Validation: Loss 0.12240 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11601 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11055</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10557</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10822</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10875</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09867</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09373</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08833</span></span>
<span class="line"><span>Validation: Loss 0.08776 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08307 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08656</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07314</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07540</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06530</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07273</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06839</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06574</span></span>
<span class="line"><span>Validation: Loss 0.06095 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05782 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06051</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05530</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05537</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04864</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04700</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04662</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04060</span></span>
<span class="line"><span>Validation: Loss 0.04495 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04271 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04399</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04247</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03719</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03917</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03678</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03971</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03252</span></span>
<span class="line"><span>Validation: Loss 0.03637 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03451 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03297</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03494</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03168</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03138</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03222</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03131</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03733</span></span>
<span class="line"><span>Validation: Loss 0.03089 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02926 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03165</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03025</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02871</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02714</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02479</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02566</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02719</span></span>
<span class="line"><span>Validation: Loss 0.02686 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02540 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02444</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02725</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02359</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02236</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02687</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02215</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02476</span></span>
<span class="line"><span>Validation: Loss 0.02377 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02245 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02161</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02260</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02225</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02342</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02007</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02045</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02144</span></span>
<span class="line"><span>Validation: Loss 0.02128 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02007 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02002</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01937</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01978</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02071</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01853</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01894</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01798</span></span>
<span class="line"><span>Validation: Loss 0.01922 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01811 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01924</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01846</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01725</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01571</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01788</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01764</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01633</span></span>
<span class="line"><span>Validation: Loss 0.01749 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01645 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01749</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01558</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01493</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01669</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01572</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01584</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01665</span></span>
<span class="line"><span>Validation: Loss 0.01601 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01504 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01531</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01526</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01408</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01418</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01513</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01441</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01478</span></span>
<span class="line"><span>Validation: Loss 0.01474 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01384 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01368</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01337</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01390</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01386</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01306</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01317</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01522</span></span>
<span class="line"><span>Validation: Loss 0.01364 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01279 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01352</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01226</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01360</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01160</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01259</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01221</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01126</span></span>
<span class="line"><span>Validation: Loss 0.01266 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01187 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01172</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01253</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01160</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01160</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01227</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01068</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01047</span></span>
<span class="line"><span>Validation: Loss 0.01178 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01104 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01146</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01187</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01025</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01066</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01139</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01058</span></span>
<span class="line"><span>Validation: Loss 0.01093 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01026 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01145</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01032</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01010</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00949</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01024</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00916</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00858</span></span>
<span class="line"><span>Validation: Loss 0.01003 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00942 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00967</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00945</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01021</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00785</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00870</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00925</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00876</span></span>
<span class="line"><span>Validation: Loss 0.00901 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00848 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00902</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00929</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00779</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00781</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00791</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00782</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00667</span></span>
<span class="line"><span>Validation: Loss 0.00801 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00756 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62686</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59189</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55786</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53801</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51328</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49906</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49657</span></span>
<span class="line"><span>Validation: Loss 0.47909 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46856 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46447</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45461</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44614</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42496</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40764</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39277</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37983</span></span>
<span class="line"><span>Validation: Loss 0.38367 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37133 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37505</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35015</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34280</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33740</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31918</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30545</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29285</span></span>
<span class="line"><span>Validation: Loss 0.29977 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28606 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29291</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27805</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25928</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25599</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23213</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23241</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21639</span></span>
<span class="line"><span>Validation: Loss 0.22957 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21572 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21074</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20613</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19065</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18171</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18299</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18283</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17709</span></span>
<span class="line"><span>Validation: Loss 0.17279 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.15992 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15913</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13994</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14437</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14647</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13224</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12985</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12269</span></span>
<span class="line"><span>Validation: Loss 0.12753 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11695 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11127</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10331</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10289</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10098</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09919</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08501</span></span>
<span class="line"><span>Validation: Loss 0.09140 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08359 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08467</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07826</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07501</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06869</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06688</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06293</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06530</span></span>
<span class="line"><span>Validation: Loss 0.06322 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05807 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05629</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05545</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05066</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05012</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04763</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04766</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04326</span></span>
<span class="line"><span>Validation: Loss 0.04666 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04308 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04389</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04210</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03645</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03910</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03769</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03806</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03123</span></span>
<span class="line"><span>Validation: Loss 0.03779 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03489 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03265</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03494</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03272</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03270</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03040</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03023</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03437</span></span>
<span class="line"><span>Validation: Loss 0.03216 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02964 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02917</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02831</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02678</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02790</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02922</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02570</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02542</span></span>
<span class="line"><span>Validation: Loss 0.02803 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02579 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02648</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02515</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02122</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02382</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02528</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02322</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02647</span></span>
<span class="line"><span>Validation: Loss 0.02483 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02281 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02164</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02345</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02228</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02193</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01929</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02163</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01850</span></span>
<span class="line"><span>Validation: Loss 0.02224 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02039 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01931</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02210</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02034</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02005</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01800</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01757</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01457</span></span>
<span class="line"><span>Validation: Loss 0.02010 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01840 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01741</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01850</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01726</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01816</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01891</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01527</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01571</span></span>
<span class="line"><span>Validation: Loss 0.01833 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01674 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01488</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01748</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01788</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01582</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01562</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01453</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01434</span></span>
<span class="line"><span>Validation: Loss 0.01682 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01533 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01549</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01545</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01429</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01438</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01463</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01420</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01265</span></span>
<span class="line"><span>Validation: Loss 0.01552 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01413 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01415</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01343</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01410</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01382</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01262</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01340</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01219</span></span>
<span class="line"><span>Validation: Loss 0.01439 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01309 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01310</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01309</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01232</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01270</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01170</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01280</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01085</span></span>
<span class="line"><span>Validation: Loss 0.01338 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01217 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01230</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01182</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01157</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01169</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01145</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01122</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01146</span></span>
<span class="line"><span>Validation: Loss 0.01244 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01131 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01166</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01033</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01107</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01032</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01042</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01103</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01143</span></span>
<span class="line"><span>Validation: Loss 0.01151 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01047 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01118</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01006</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01062</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00881</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00999</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00856</span></span>
<span class="line"><span>Validation: Loss 0.01046 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00953 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00899</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00875</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00960</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00964</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00847</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00865</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00893</span></span>
<span class="line"><span>Validation: Loss 0.00931 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00851 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00861</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00737</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00861</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00735</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00812</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00828</span></span>
<span class="line"><span>Validation: Loss 0.00829 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00760 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>- CUDA_Runtime_jll: 0.15.3+0</span></span>
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
