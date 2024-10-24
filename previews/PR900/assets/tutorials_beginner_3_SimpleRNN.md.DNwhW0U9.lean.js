import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.D4MVDo3k.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR900/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR900/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR900/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR900/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62077</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59964</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56938</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53396</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51702</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50271</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49064</span></span>
<span class="line"><span>Validation: Loss 0.46925 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.45939 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47154</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45123</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45105</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42202</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40880</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39616</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37961</span></span>
<span class="line"><span>Validation: Loss 0.37195 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36037 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36403</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35401</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34694</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33633</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32301</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31097</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31633</span></span>
<span class="line"><span>Validation: Loss 0.28688 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.27372 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28813</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26548</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27612</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25184</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24357</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23556</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22515</span></span>
<span class="line"><span>Validation: Loss 0.21689 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.20353 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21203</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21583</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19829</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18468</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17388</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18807</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16192</span></span>
<span class="line"><span>Validation: Loss 0.16109 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.14862 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15655</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15557</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16250</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14597</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12608</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12145</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11854</span></span>
<span class="line"><span>Validation: Loss 0.11813 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10774 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11711</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11882</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11046</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11189</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09625</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08251</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07253</span></span>
<span class="line"><span>Validation: Loss 0.08473 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07696 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08613</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08366</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07053</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07547</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06910</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06167</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07042</span></span>
<span class="line"><span>Validation: Loss 0.05925 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05403 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06466</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05406</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05280</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04875</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04843</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05016</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03622</span></span>
<span class="line"><span>Validation: Loss 0.04374 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04015 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04305</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04395</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04385</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03542</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04058</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03400</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03956</span></span>
<span class="line"><span>Validation: Loss 0.03531 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03243 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03418</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03216</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03466</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03279</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03622</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02845</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03137</span></span>
<span class="line"><span>Validation: Loss 0.02994 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02745 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03166</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02883</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02895</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02719</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02735</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02592</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02809</span></span>
<span class="line"><span>Validation: Loss 0.02603 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02383 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02583</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02535</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02316</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02506</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02422</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02443</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02644</span></span>
<span class="line"><span>Validation: Loss 0.02302 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02103 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02400</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02362</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02014</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02211</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02194</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02029</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02147</span></span>
<span class="line"><span>Validation: Loss 0.02058 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01877 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01985</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02027</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01872</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02144</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01903</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01946</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01799</span></span>
<span class="line"><span>Validation: Loss 0.01857 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01690 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01856</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01848</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01757</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01878</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01694</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01717</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01630</span></span>
<span class="line"><span>Validation: Loss 0.01688 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01533 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01564</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01697</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01690</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01683</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01514</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01656</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01432</span></span>
<span class="line"><span>Validation: Loss 0.01544 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01399 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01559</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01612</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01579</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01404</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01415</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01468</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01149</span></span>
<span class="line"><span>Validation: Loss 0.01421 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01285 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01462</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01472</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01336</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01346</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01397</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01279</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01215</span></span>
<span class="line"><span>Validation: Loss 0.01314 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01187 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01315</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01313</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01312</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01161</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01270</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01257</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01316</span></span>
<span class="line"><span>Validation: Loss 0.01219 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01100 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01147</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01251</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01227</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01315</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01176</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00953</span></span>
<span class="line"><span>Validation: Loss 0.01128 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01018 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01173</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01030</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01114</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01037</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01150</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01080</span></span>
<span class="line"><span>Validation: Loss 0.01035 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00934 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00974</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01020</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01008</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01059</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00957</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00933</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01011</span></span>
<span class="line"><span>Validation: Loss 0.00929 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00840 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00981</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00928</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00831</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00867</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00806</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00920</span></span>
<span class="line"><span>Validation: Loss 0.00823 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00748 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00787</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00739</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00814</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00795</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00818</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00800</span></span>
<span class="line"><span>Validation: Loss 0.00743 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00678 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62700</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59137</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.56343</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54346</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52579</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.49035</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47167</span></span>
<span class="line"><span>Validation: Loss 0.47291 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46865 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46939</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44936</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44308</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42121</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41285</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39532</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39444</span></span>
<span class="line"><span>Validation: Loss 0.37665 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37126 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37512</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36478</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33763</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33490</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30874</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31622</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.28148</span></span>
<span class="line"><span>Validation: Loss 0.29192 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28619 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27660</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28160</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25287</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26117</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25023</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23170</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23068</span></span>
<span class="line"><span>Validation: Loss 0.22185 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21627 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21442</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19402</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21303</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18680</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18025</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17646</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17251</span></span>
<span class="line"><span>Validation: Loss 0.16544 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16052 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16546</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15476</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13869</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14173</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13468</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12768</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.11424</span></span>
<span class="line"><span>Validation: Loss 0.12131 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11740 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11556</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10893</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10520</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10312</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10139</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08850</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09472</span></span>
<span class="line"><span>Validation: Loss 0.08675 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08391 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08304</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07316</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07511</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07143</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06865</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06881</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06155</span></span>
<span class="line"><span>Validation: Loss 0.06028 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05841 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05599</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05380</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05399</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05033</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04953</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04750</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04500</span></span>
<span class="line"><span>Validation: Loss 0.04486 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04351 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04347</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04442</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04203</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03788</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03830</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03353</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03862</span></span>
<span class="line"><span>Validation: Loss 0.03638 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03525 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03814</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03265</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03464</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03272</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03038</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03072</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02657</span></span>
<span class="line"><span>Validation: Loss 0.03092 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02993 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02870</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02808</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02728</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02911</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02878</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02753</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02748</span></span>
<span class="line"><span>Validation: Loss 0.02697 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02608 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02384</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02382</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02579</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02514</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02430</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02536</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02516</span></span>
<span class="line"><span>Validation: Loss 0.02389 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02309 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02126</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02089</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02313</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02373</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02089</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02249</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02024</span></span>
<span class="line"><span>Validation: Loss 0.02140 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02067 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02104</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02144</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02004</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01850</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01877</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01920</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01826</span></span>
<span class="line"><span>Validation: Loss 0.01933 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01865 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01848</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01822</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01813</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01657</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01798</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01726</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.02039</span></span>
<span class="line"><span>Validation: Loss 0.01759 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01696 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01759</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01673</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01669</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01629</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01396</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01605</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01809</span></span>
<span class="line"><span>Validation: Loss 0.01608 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01550 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01594</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01457</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01483</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01594</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01420</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01430</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01386</span></span>
<span class="line"><span>Validation: Loss 0.01479 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01425 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01328</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01381</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01487</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01431</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01321</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01320</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01280</span></span>
<span class="line"><span>Validation: Loss 0.01368 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01317 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01368</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01290</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01376</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01213</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01187</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01234</span></span>
<span class="line"><span>Validation: Loss 0.01269 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01222 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01257</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01113</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01232</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01161</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01218</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01124</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01099</span></span>
<span class="line"><span>Validation: Loss 0.01178 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01134 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01044</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01232</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01152</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01023</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01053</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01010</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01313</span></span>
<span class="line"><span>Validation: Loss 0.01087 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01047 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01007</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01073</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01030</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00948</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01031</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00969</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00891</span></span>
<span class="line"><span>Validation: Loss 0.00985 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00949 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00904</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00943</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00929</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00794</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00941</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00916</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00894</span></span>
<span class="line"><span>Validation: Loss 0.00876 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00846 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00879</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00871</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00764</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00754</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00778</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00829</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00657</span></span>
<span class="line"><span>Validation: Loss 0.00784 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00758 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.422 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
