import{_ as a,o as n,c as i,a2 as p}from"./chunks/framework.DgNEeqZX.js";const r=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),l={name:"tutorials/beginner/3_SimpleRNN.md"};function e(h,s,t,c,k,o){return n(),i("div",null,s[0]||(s[0]=[p(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random, Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractLuxContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v1.3.2/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Lux.AbstractLuxContainerLayer{(:lstm_cell, :classifier)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v1.3.2/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v1.3.2/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v1.3.2/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.63025</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.59252</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57088</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52932</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51400</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50206</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48769</span></span>
<span class="line"><span>Validation: Loss 0.47018 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.47064 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46553</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45533</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43983</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.42447</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41724</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39576</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36962</span></span>
<span class="line"><span>Validation: Loss 0.37282 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37357 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37650</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37466</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34152</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34363</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30602</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29191</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.29158</span></span>
<span class="line"><span>Validation: Loss 0.28819 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.28900 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27525</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27721</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.26404</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25525</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24492</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.23775</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24253</span></span>
<span class="line"><span>Validation: Loss 0.21888 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.21954 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20728</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20676</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20946</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20387</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18062</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17109</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.14328</span></span>
<span class="line"><span>Validation: Loss 0.16350 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.16388 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16040</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.16014</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14114</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13844</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13959</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12868</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12854</span></span>
<span class="line"><span>Validation: Loss 0.12064 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12080 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11828</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11069</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11143</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10270</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09966</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09235</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.08941</span></span>
<span class="line"><span>Validation: Loss 0.08659 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.08664 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08308</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08126</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07705</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07120</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06869</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06998</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05573</span></span>
<span class="line"><span>Validation: Loss 0.06021 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06021 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05737</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05734</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05609</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05445</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04675</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04521</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04244</span></span>
<span class="line"><span>Validation: Loss 0.04441 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04440 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04378</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03780</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04223</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04091</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03903</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03821</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03259</span></span>
<span class="line"><span>Validation: Loss 0.03584 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03582 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03564</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03466</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03382</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03248</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03093</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03074</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03161</span></span>
<span class="line"><span>Validation: Loss 0.03040 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03038 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02863</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03059</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02746</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02981</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02637</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02821</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02189</span></span>
<span class="line"><span>Validation: Loss 0.02644 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02642 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02332</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02555</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02465</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02553</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02551</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02360</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02509</span></span>
<span class="line"><span>Validation: Loss 0.02341 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02338 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02241</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02304</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02248</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02308</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02177</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01971</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01938</span></span>
<span class="line"><span>Validation: Loss 0.02094 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02092 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02039</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02046</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02162</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01877</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02046</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01700</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01815</span></span>
<span class="line"><span>Validation: Loss 0.01890 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01888 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01942</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01780</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01756</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01782</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01808</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01651</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01712</span></span>
<span class="line"><span>Validation: Loss 0.01718 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01717 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01608</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01730</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01582</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01500</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01773</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01523</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01745</span></span>
<span class="line"><span>Validation: Loss 0.01573 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01571 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01560</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01450</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01537</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01561</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01471</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01412</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01268</span></span>
<span class="line"><span>Validation: Loss 0.01447 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01445 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01393</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01405</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01335</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01373</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01399</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01357</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01277</span></span>
<span class="line"><span>Validation: Loss 0.01340 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01338 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01249</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01340</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01339</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01211</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01323</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01197</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01188</span></span>
<span class="line"><span>Validation: Loss 0.01245 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01243 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01186</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01197</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01257</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01055</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01220</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01243</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00968</span></span>
<span class="line"><span>Validation: Loss 0.01160 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01159 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01115</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01105</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01128</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01072</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01182</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01029</span></span>
<span class="line"><span>Validation: Loss 0.01081 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01080 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01030</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01114</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01071</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01069</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00957</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00966</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00875</span></span>
<span class="line"><span>Validation: Loss 0.01002 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01001 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01013</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00904</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00941</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00929</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00982</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00862</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.01167</span></span>
<span class="line"><span>Validation: Loss 0.00914 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00913 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00862</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00834</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00848</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00881</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00826</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00885</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00872</span></span>
<span class="line"><span>Validation: Loss 0.00815 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00814 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.62511</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.58580</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.57456</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.53250</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.52901</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50098</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48846</span></span>
<span class="line"><span>Validation: Loss 0.47042 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.45265 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.46477</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.45467</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.44332</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.43183</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.41632</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39695</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.39054</span></span>
<span class="line"><span>Validation: Loss 0.37357 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.35305 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.36736</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.37108</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.34727</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32543</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.32855</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.30059</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.33959</span></span>
<span class="line"><span>Validation: Loss 0.28889 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26646 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27500</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.28522</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.27535</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24723</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.25147</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.24482</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.21234</span></span>
<span class="line"><span>Validation: Loss 0.21940 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.19670 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21717</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.21005</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.20097</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.18944</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.19837</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.17691</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.15471</span></span>
<span class="line"><span>Validation: Loss 0.16378 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.14238 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14907</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.17096</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.14491</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.15415</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.12732</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13610</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.13404</span></span>
<span class="line"><span>Validation: Loss 0.12086 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.10280 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.11218</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.12446</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10434</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10600</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10753</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.09145</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.10018</span></span>
<span class="line"><span>Validation: Loss 0.08695 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.07342 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08309</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08118</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.08338</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07706</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.07309</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.06513</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05652</span></span>
<span class="line"><span>Validation: Loss 0.06052 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05150 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.06397</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05313</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05611</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05228</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04761</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.05207</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.04120</span></span>
<span class="line"><span>Validation: Loss 0.04462 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03834 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04387</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04480</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04296</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.04029</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03845</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03702</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03665</span></span>
<span class="line"><span>Validation: Loss 0.03597 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03090 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03239</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03502</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03643</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03282</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03230</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03393</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.03191</span></span>
<span class="line"><span>Validation: Loss 0.03050 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02608 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03072</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02933</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02973</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02618</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.03022</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02776</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02768</span></span>
<span class="line"><span>Validation: Loss 0.02652 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02260 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02927</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02735</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02444</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02418</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02486</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02272</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.02274</span></span>
<span class="line"><span>Validation: Loss 0.02343 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01990 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02363</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02282</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02408</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02229</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02117</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02103</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.02261</span></span>
<span class="line"><span>Validation: Loss 0.02097 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01775 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02039</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02010</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01996</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02132</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02028</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01912</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.02045</span></span>
<span class="line"><span>Validation: Loss 0.01894 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01597 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01857</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01935</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01823</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01933</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01691</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01766</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01731</span></span>
<span class="line"><span>Validation: Loss 0.01721 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01446 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01735</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01699</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01565</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01639</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01649</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01720</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01640</span></span>
<span class="line"><span>Validation: Loss 0.01574 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01318 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01693</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01508</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01462</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01488</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01659</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01439</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.01220</span></span>
<span class="line"><span>Validation: Loss 0.01448 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01208 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01602</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01445</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01387</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01343</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01328</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01381</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.01286</span></span>
<span class="line"><span>Validation: Loss 0.01339 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01114 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01352</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01324</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01399</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01270</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01193</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01294</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.01291</span></span>
<span class="line"><span>Validation: Loss 0.01242 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01032 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01290</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01180</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01179</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01227</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01156</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01248</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.01153</span></span>
<span class="line"><span>Validation: Loss 0.01153 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00958 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01096</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01038</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01275</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01049</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01147</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.01167</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00984</span></span>
<span class="line"><span>Validation: Loss 0.01065 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00885 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01022</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01032</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01107</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.01110</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00922</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00974</span></span>
<span class="line"><span>Validation: Loss 0.00968 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00807 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00985</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00937</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00896</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00885</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00926</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00871</span></span>
<span class="line"><span>Validation: Loss 0.00861 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00722 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00871</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00854</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00778</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00872</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00788</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00801</span></span>
<span class="line"><span>Validation: Loss 0.00768 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00649 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model struct and only save the parameters and states.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
