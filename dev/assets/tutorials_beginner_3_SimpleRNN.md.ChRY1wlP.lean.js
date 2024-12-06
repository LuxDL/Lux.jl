import{_ as a,c as n,a2 as i,o as p}from"./chunks/framework.BS99Di-t.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return p(),n("div",null,s[0]||(s[0]=[i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/dev/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/dev/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/dev/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##230&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/dev/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62825</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57987</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55511</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54567</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51745</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50528</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48849</span></span>
<span class="line"><span>Validation: Loss 0.47171 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47835 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46148</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45396</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43202</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41976</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42554</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.40721</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37973</span></span>
<span class="line"><span>Validation: Loss 0.37538 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.38334 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37401</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36223</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33914</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32716</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31866</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31896</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29863</span></span>
<span class="line"><span>Validation: Loss 0.29095 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.29941 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29520</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27378</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26995</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26258</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22761</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.22629</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25562</span></span>
<span class="line"><span>Validation: Loss 0.22083 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.22894 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20594</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19640</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20504</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19794</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19364</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.16974</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18443</span></span>
<span class="line"><span>Validation: Loss 0.16476 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17182 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15971</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15001</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16084</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14325</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13378</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12436</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.10739</span></span>
<span class="line"><span>Validation: Loss 0.12102 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12655 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12102</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11160</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10138</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10264</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09948</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09989</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06973</span></span>
<span class="line"><span>Validation: Loss 0.08676 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09077 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08504</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07378</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07145</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07636</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06756</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07041</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06700</span></span>
<span class="line"><span>Validation: Loss 0.06052 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06321 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05634</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05263</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05554</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05161</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04959</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04877</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04893</span></span>
<span class="line"><span>Validation: Loss 0.04502 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04697 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04794</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04099</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03903</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03977</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03794</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03787</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03681</span></span>
<span class="line"><span>Validation: Loss 0.03647 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03806 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03680</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03406</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03669</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03189</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03029</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03234</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02680</span></span>
<span class="line"><span>Validation: Loss 0.03098 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03235 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03125</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02877</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02894</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02939</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02757</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02583</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02894</span></span>
<span class="line"><span>Validation: Loss 0.02698 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02821 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02508</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02763</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02783</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02468</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02378</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02208</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02298</span></span>
<span class="line"><span>Validation: Loss 0.02387 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02497 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02365</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02201</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02206</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02233</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02137</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02223</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02209</span></span>
<span class="line"><span>Validation: Loss 0.02137 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02237 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02311</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01990</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01997</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02043</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01846</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01928</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01558</span></span>
<span class="line"><span>Validation: Loss 0.01929 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02021 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01838</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01778</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01877</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01781</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01826</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01734</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01881</span></span>
<span class="line"><span>Validation: Loss 0.01755 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01841 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01671</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01698</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01826</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01607</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01635</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01551</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01283</span></span>
<span class="line"><span>Validation: Loss 0.01606 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01686 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01683</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01586</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01249</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01444</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01561</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01589</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01351</span></span>
<span class="line"><span>Validation: Loss 0.01480 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01554 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01495</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01449</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01376</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01346</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01346</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01330</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01520</span></span>
<span class="line"><span>Validation: Loss 0.01369 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01438 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01423</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01314</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01339</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01233</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01272</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01207</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01155</span></span>
<span class="line"><span>Validation: Loss 0.01268 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01332 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01243</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01180</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01240</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01265</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01129</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01140</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01118</span></span>
<span class="line"><span>Validation: Loss 0.01173 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01231 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01148</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01034</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01111</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01109</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01073</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01115</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01190</span></span>
<span class="line"><span>Validation: Loss 0.01072 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01125 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00917</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01037</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01010</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00954</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01037</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00999</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01140</span></span>
<span class="line"><span>Validation: Loss 0.00957 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01002 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00855</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00950</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00837</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00891</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00949</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00860</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00811</span></span>
<span class="line"><span>Validation: Loss 0.00847 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00886 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00816</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00819</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00807</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00735</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00805</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00779</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00765</span></span>
<span class="line"><span>Validation: Loss 0.00767 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00802 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.61962</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57844</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.55840</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.54874</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52611</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50168</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50341</span></span>
<span class="line"><span>Validation: Loss 0.47373 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.46774 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.47020</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44898</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43600</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42631</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43243</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39552</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38203</span></span>
<span class="line"><span>Validation: Loss 0.37833 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37154 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37584</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35243</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.35176</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33528</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32299</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31479</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.31218</span></span>
<span class="line"><span>Validation: Loss 0.29457 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28732 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.29178</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28059</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27065</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26285</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24416</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23447</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21863</span></span>
<span class="line"><span>Validation: Loss 0.22470 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21759 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.23153</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20307</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20587</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18230</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18828</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18604</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15516</span></span>
<span class="line"><span>Validation: Loss 0.16840 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16195 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16171</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16423</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15477</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14540</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13710</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12244</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13109</span></span>
<span class="line"><span>Validation: Loss 0.12443 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.11921 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11049</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11783</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11355</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10569</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10462</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10048</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07178</span></span>
<span class="line"><span>Validation: Loss 0.08955 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08573 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08967</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07913</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07737</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08036</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06850</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06550</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06645</span></span>
<span class="line"><span>Validation: Loss 0.06234 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05979 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06026</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06021</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05458</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04897</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05297</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04656</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04799</span></span>
<span class="line"><span>Validation: Loss 0.04588 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04407 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04098</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04358</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04433</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03850</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03928</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04013</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03705</span></span>
<span class="line"><span>Validation: Loss 0.03704 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03556 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03731</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03369</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03609</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03177</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03245</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03117</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03450</span></span>
<span class="line"><span>Validation: Loss 0.03139 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03011 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03139</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02889</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02752</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02875</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02817</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02810</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03096</span></span>
<span class="line"><span>Validation: Loss 0.02729 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02616 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02729</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02655</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02700</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02399</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02381</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02341</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02385</span></span>
<span class="line"><span>Validation: Loss 0.02410 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02308 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02220</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02307</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02269</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02192</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02359</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02184</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01897</span></span>
<span class="line"><span>Validation: Loss 0.02156 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02063 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02311</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02086</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01923</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02023</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01791</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01966</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01920</span></span>
<span class="line"><span>Validation: Loss 0.01947 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01861 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01840</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01813</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01718</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01815</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01963</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01819</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01609</span></span>
<span class="line"><span>Validation: Loss 0.01771 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01692 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01556</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01763</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01596</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01737</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01652</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01693</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01471</span></span>
<span class="line"><span>Validation: Loss 0.01622 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01548 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01730</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01433</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01477</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01594</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01500</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01410</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01467</span></span>
<span class="line"><span>Validation: Loss 0.01493 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01424 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01507</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01361</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01361</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01471</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01260</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01494</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01246</span></span>
<span class="line"><span>Validation: Loss 0.01381 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01317 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01232</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01393</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01379</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01167</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01322</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01309</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01257</span></span>
<span class="line"><span>Validation: Loss 0.01281 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01221 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01307</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01043</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01328</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01228</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01117</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01203</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01215</span></span>
<span class="line"><span>Validation: Loss 0.01187 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01132 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01199</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01081</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01149</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01046</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01088</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01121</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01117</span></span>
<span class="line"><span>Validation: Loss 0.01090 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01040 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01090</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00992</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00965</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01022</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00975</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01085</span></span>
<span class="line"><span>Validation: Loss 0.00980 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00937 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00908</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00975</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01009</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00879</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00852</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00829</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00887</span></span>
<span class="line"><span>Validation: Loss 0.00866 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00828 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00826</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00806</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00831</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00772</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00834</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00779</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00737</span></span>
<span class="line"><span>Validation: Loss 0.00778 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00745 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.11.2</span></span>
<span class="line"><span>Commit 5e9a32e7af2 (2024-12-01 20:02 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LLVM: libLLVM-16.0.6 (ORCJIT, znver2)</span></span>
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
<span class="line"><span>- CUBLAS: 12.6.4</span></span>
<span class="line"><span>- CURAND: 10.3.7</span></span>
<span class="line"><span>- CUFFT: 11.3.0</span></span>
<span class="line"><span>- CUSOLVER: 11.7.1</span></span>
<span class="line"><span>- CUSPARSE: 12.5.4</span></span>
<span class="line"><span>- CUPTI: 2024.3.2 (API 24.0.0)</span></span>
<span class="line"><span>- NVML: 12.0.0+560.35.3</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.5.2</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.10.4+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.15.5+0</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.11.2</span></span>
<span class="line"><span>- LLVM: 16.0.6</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: Quadro RTX 5000 (sm_75, 14.915 GiB / 16.000 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45)]))}const d=a(l,[["render",e]]);export{r as __pageData,d as default};
