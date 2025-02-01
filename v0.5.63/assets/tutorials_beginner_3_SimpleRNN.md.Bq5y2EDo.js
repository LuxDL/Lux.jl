import{_ as s,c as a,o as n,a4 as i}from"./chunks/framework.BZXF45WW.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">      Statistics</span></span></code></pre></div><h2 id="dataset" tabindex="-1">Dataset <a class="header-anchor" href="#dataset" aria-label="Permalink to &quot;Dataset&quot;">​</a></h2><p>We will use MLUtils to generate 500 (noisy) clockwise and 500 (noisy) anticlockwise spirals. Using this data we will create a <code>MLUtils.DataLoader</code>. Our dataloader will give us sequences of size 2 × seq_len × batch_size and we need to predict a binary value whether the sequence is clockwise or anticlockwise.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; dataset_size</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, sequence_length</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">50</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v0.5.63/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v0.5.63/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v0.5.63/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Main.var&quot;##225&quot;.SpiralClassifier</span></span></code></pre></div><p>We can use default Lux blocks – <code>Recurrence(LSTMCell(in_dims =&gt; hidden_dims)</code> – instead of defining the following. But let&#39;s still do it for the sake of it.</p><p>Now we need to define the behavior of the Classifier when it is invoked.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (s</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ps</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, st</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # First we will have to run the sequence through the LSTM Cell</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # The first call to LSTM Cell will create the initial hidden state</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # See that the parameters and states are automatically populated into a field called</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # \`lstm_cell\` We use \`eachslice\` to get the elements in the sequence without copying,</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # and \`Iterators.peel\` to split out the first element for LSTM initialization.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v0.5.63/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> LSTMCell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> hidden_dims)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Dense</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(hidden_dims </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> out_dims, sigmoid)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @compact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; lstm_cell, classifier) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">do</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{T, 3}</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        x_init, x_rest </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Iterators</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">peel</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">_eachslice</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Val</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
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
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Get the dataloaders</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    (train_loader, val_loader) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> get_dataloaders</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Create the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model_type</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    rng </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Xoshiro</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    dev </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> gpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(rng, model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); transform_variables</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dev)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Training</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), lossfn, (x, y), train_state)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56108</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50589</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47227</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45056</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42777</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40461</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38121</span></span>
<span class="line"><span>Validation: Loss 0.37370 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36046 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36482</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35187</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33582</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31633</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29969</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28001</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27801</span></span>
<span class="line"><span>Validation: Loss 0.26019 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25246 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25553</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24669</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22788</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22091</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20665</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19928</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18737</span></span>
<span class="line"><span>Validation: Loss 0.18029 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17661 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18152</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16906</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16260</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15094</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14572</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14147</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13368</span></span>
<span class="line"><span>Validation: Loss 0.12825 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12585 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12783</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12172</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11602</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10990</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10584</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10063</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09586</span></span>
<span class="line"><span>Validation: Loss 0.09337 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09105 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09227</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08767</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08452</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08107</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07848</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07362</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.06966</span></span>
<span class="line"><span>Validation: Loss 0.06922 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06667 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06951</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06578</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06288</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05697</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05745</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05591</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05138</span></span>
<span class="line"><span>Validation: Loss 0.05197 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04941 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05141</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04903</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04641</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04415</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04249</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04145</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04204</span></span>
<span class="line"><span>Validation: Loss 0.03941 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03697 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03873</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03660</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03504</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03451</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03260</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03120</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02909</span></span>
<span class="line"><span>Validation: Loss 0.03031 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02809 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02868</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02843</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02778</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02549</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02540</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02462</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02484</span></span>
<span class="line"><span>Validation: Loss 0.02396 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02200 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02235</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02356</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02213</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02088</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01957</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01893</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02135</span></span>
<span class="line"><span>Validation: Loss 0.01953 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01786 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01809</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01735</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01859</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01771</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01683</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01675</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01559</span></span>
<span class="line"><span>Validation: Loss 0.01643 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01499 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01626</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01508</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01499</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01434</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01492</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01367</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01419</span></span>
<span class="line"><span>Validation: Loss 0.01419 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01293 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01406</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01305</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01271</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01286</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01245</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01283</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01122</span></span>
<span class="line"><span>Validation: Loss 0.01251 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01139 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01244</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01165</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01154</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01167</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01159</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01028</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01001</span></span>
<span class="line"><span>Validation: Loss 0.01121 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01020 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01007</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01076</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00976</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01013</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01067</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01034</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01066</span></span>
<span class="line"><span>Validation: Loss 0.01016 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00923 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00986</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00970</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00926</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00936</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00906</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00892</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00948</span></span>
<span class="line"><span>Validation: Loss 0.00927 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00842 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00887</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00857</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00879</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00844</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00846</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00866</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00705</span></span>
<span class="line"><span>Validation: Loss 0.00852 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00773 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00765</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00836</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00793</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00757</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00766</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00763</span></span>
<span class="line"><span>Validation: Loss 0.00788 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00714 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00750</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00773</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00722</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00740</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00702</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00704</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00705</span></span>
<span class="line"><span>Validation: Loss 0.00732 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00663 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00727</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00704</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00664</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00688</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00721</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00678</span></span>
<span class="line"><span>Validation: Loss 0.00683 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00618 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00680</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00681</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00636</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00600</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00575</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00635</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00634</span></span>
<span class="line"><span>Validation: Loss 0.00639 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00578 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00616</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00603</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00578</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00577</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00580</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00620</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00558</span></span>
<span class="line"><span>Validation: Loss 0.00600 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00542 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00566</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00552</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00566</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00555</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00546</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00511</span></span>
<span class="line"><span>Validation: Loss 0.00564 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00510 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00565</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00514</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00511</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00540</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00506</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00533</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00465</span></span>
<span class="line"><span>Validation: Loss 0.00533 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00481 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56313</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51335</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47197</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44741</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42450</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40791</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40046</span></span>
<span class="line"><span>Validation: Loss 0.35891 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36773 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36975</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35492</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.32942</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31546</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30027</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28768</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27563</span></span>
<span class="line"><span>Validation: Loss 0.25284 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25773 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25706</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24294</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23153</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21894</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21080</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19947</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19368</span></span>
<span class="line"><span>Validation: Loss 0.17759 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17974 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17917</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17142</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16218</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15494</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14748</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14068</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13370</span></span>
<span class="line"><span>Validation: Loss 0.12699 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12812 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12806</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12161</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11607</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11122</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10692</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10175</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09803</span></span>
<span class="line"><span>Validation: Loss 0.09155 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09291 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09236</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08870</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08556</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08147</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07696</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07515</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07242</span></span>
<span class="line"><span>Validation: Loss 0.06680 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06838 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06807</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06581</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06323</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05929</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05833</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05612</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05175</span></span>
<span class="line"><span>Validation: Loss 0.04927 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05091 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04907</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04931</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04918</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04394</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04289</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04298</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03966</span></span>
<span class="line"><span>Validation: Loss 0.03663 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03825 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03878</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03738</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03553</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03427</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03204</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03202</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02959</span></span>
<span class="line"><span>Validation: Loss 0.02770 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02922 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02871</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02939</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02610</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02857</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02481</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02467</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02326</span></span>
<span class="line"><span>Validation: Loss 0.02163 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02300 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02231</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02322</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02174</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02112</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02135</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02005</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01735</span></span>
<span class="line"><span>Validation: Loss 0.01754 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01874 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01952</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01740</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01746</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01801</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01745</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01715</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01483</span></span>
<span class="line"><span>Validation: Loss 0.01472 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01577 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01565</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01615</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01503</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01427</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01545</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01475</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01157</span></span>
<span class="line"><span>Validation: Loss 0.01270 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01363 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01399</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01402</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01276</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01258</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01303</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01262</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01259</span></span>
<span class="line"><span>Validation: Loss 0.01120 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01204 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01220</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01151</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01189</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01201</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01182</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01091</span></span>
<span class="line"><span>Validation: Loss 0.01002 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01079 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01083</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01070</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01070</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01064</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01084</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00945</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00991</span></span>
<span class="line"><span>Validation: Loss 0.00907 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00976 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00983</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01005</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00967</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00917</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00967</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00929</span></span>
<span class="line"><span>Validation: Loss 0.00827 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00892 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00861</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00823</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00838</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00920</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00856</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00855</span></span>
<span class="line"><span>Validation: Loss 0.00760 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00820 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00855</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00827</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00846</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00792</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00721</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00813</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00699</span></span>
<span class="line"><span>Validation: Loss 0.00702 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00757 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00738</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00765</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00753</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00717</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00741</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00732</span></span>
<span class="line"><span>Validation: Loss 0.00651 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00703 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00700</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00679</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00701</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00674</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00674</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00583</span></span>
<span class="line"><span>Validation: Loss 0.00607 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00655 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00679</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00674</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00615</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00606</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00669</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00653</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00583</span></span>
<span class="line"><span>Validation: Loss 0.00567 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00613 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00581</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00625</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00599</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00649</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00605</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00595</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00527</span></span>
<span class="line"><span>Validation: Loss 0.00532 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00576 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00568</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00596</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00549</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00585</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00556</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00560</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00577</span></span>
<span class="line"><span>Validation: Loss 0.00501 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00542 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00534</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00534</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00540</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00572</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00492</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00544</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00546</span></span>
<span class="line"><span>Validation: Loss 0.00472 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00512 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
<span class="line"><span> :ps_trained</span></span>
<span class="line"><span> :st_trained</span></span></code></pre></div><h2 id="appendix" tabindex="-1">Appendix <a class="header-anchor" href="#appendix" aria-label="Permalink to &quot;Appendix&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> InteractiveUtils</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">InteractiveUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxDeviceUtils)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(CUDA) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxCUDADevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        CUDA</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    if</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @isdefined</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(AMDGPU) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&amp;&amp;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> LuxDeviceUtils</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">functional</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(LuxAMDGPUDevice)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">        println</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        AMDGPU</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">versioninfo</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Julia Version 1.10.4</span></span>
<span class="line"><span>Commit 48d4fd48430 (2024-06-04 10:41 UTC)</span></span>
<span class="line"><span>Build Info:</span></span>
<span class="line"><span>  Official https://julialang.org/ release</span></span>
<span class="line"><span>Platform Info:</span></span>
<span class="line"><span>  OS: Linux (x86_64-linux-gnu)</span></span>
<span class="line"><span>  CPU: 48 × AMD EPYC 7402 24-Core Processor</span></span>
<span class="line"><span>  WORD_SIZE: 64</span></span>
<span class="line"><span>  LIBM: libopenlibm</span></span>
<span class="line"><span>  LLVM: libLLVM-15.0.7 (ORCJIT, znver2)</span></span>
<span class="line"><span>Threads: 4 default, 0 interactive, 2 GC (on 2 virtual cores)</span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>  JULIA_CPU_THREADS = 2</span></span>
<span class="line"><span>  JULIA_DEPOT_PATH = /root/.cache/julia-buildkite-plugin/depots/01872db4-8c79-43af-ab7d-12abac4f24f6</span></span>
<span class="line"><span>  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64</span></span>
<span class="line"><span>  JULIA_PKG_SERVER = </span></span>
<span class="line"><span>  JULIA_NUM_THREADS = 4</span></span>
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
<span class="line"><span>- Julia: 1.10.4</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,c,k,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
