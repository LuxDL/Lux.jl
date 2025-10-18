import{_ as s,c as a,o as n,a4 as i}from"./chunks/framework.C_hFR9fe.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v0.5.65/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v0.5.65/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v0.5.65/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v0.5.65/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.55994</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51047</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48322</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44852</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.41908</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40029</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40570</span></span>
<span class="line"><span>Validation: Loss 0.36805 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36347 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36024</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36134</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33199</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31666</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30262</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28749</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27449</span></span>
<span class="line"><span>Validation: Loss 0.25828 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25599 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25749</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24880</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23370</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22128</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21029</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19966</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19183</span></span>
<span class="line"><span>Validation: Loss 0.18054 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17970 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18303</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17026</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16181</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15543</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14966</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14363</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14015</span></span>
<span class="line"><span>Validation: Loss 0.12906 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12858 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12910</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12285</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11793</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11315</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10810</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10394</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09901</span></span>
<span class="line"><span>Validation: Loss 0.09403 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09352 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09613</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09010</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08369</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08278</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08052</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07641</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07043</span></span>
<span class="line"><span>Validation: Loss 0.06942 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06877 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06958</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06656</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06450</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06088</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05907</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05713</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05341</span></span>
<span class="line"><span>Validation: Loss 0.05180 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05102 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05416</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05140</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04721</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04507</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04158</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04248</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04365</span></span>
<span class="line"><span>Validation: Loss 0.03900 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03818 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03816</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03725</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03583</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03687</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03341</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03238</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03037</span></span>
<span class="line"><span>Validation: Loss 0.02984 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02904 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02955</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02806</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02834</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02977</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02615</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02378</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02252</span></span>
<span class="line"><span>Validation: Loss 0.02353 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02280 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02311</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02404</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02285</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02123</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02148</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01949</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01811</span></span>
<span class="line"><span>Validation: Loss 0.01918 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01853 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02028</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01769</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01807</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01881</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01758</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01637</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01608</span></span>
<span class="line"><span>Validation: Loss 0.01615 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01558 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01741</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01555</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01513</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01544</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01442</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01464</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01343</span></span>
<span class="line"><span>Validation: Loss 0.01396 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01346 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01450</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01401</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01375</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01333</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01325</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01204</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01107</span></span>
<span class="line"><span>Validation: Loss 0.01231 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01187 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01335</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01233</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01220</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01182</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01112</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01102</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.00963</span></span>
<span class="line"><span>Validation: Loss 0.01103 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01063 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01145</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01024</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01194</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00970</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00988</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01076</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01113</span></span>
<span class="line"><span>Validation: Loss 0.00999 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00962 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01009</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01004</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01018</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00968</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00900</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00924</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00983</span></span>
<span class="line"><span>Validation: Loss 0.00912 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00878 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00939</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00865</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00864</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00885</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00837</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00934</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00899</span></span>
<span class="line"><span>Validation: Loss 0.00838 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00806 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00903</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00843</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00825</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00742</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00789</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00803</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00830</span></span>
<span class="line"><span>Validation: Loss 0.00774 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00744 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00714</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00782</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00714</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00745</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00709</span></span>
<span class="line"><span>Validation: Loss 0.00718 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00691 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00720</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00719</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00723</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00691</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00706</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00679</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00635</span></span>
<span class="line"><span>Validation: Loss 0.00669 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00644 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00648</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00668</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00643</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00689</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00650</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00641</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00651</span></span>
<span class="line"><span>Validation: Loss 0.00626 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00602 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00622</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00612</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00632</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00596</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00632</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00594</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00615</span></span>
<span class="line"><span>Validation: Loss 0.00588 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00565 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00574</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00573</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00579</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00580</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00586</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00591</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00507</span></span>
<span class="line"><span>Validation: Loss 0.00553 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00531 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00511</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00567</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00562</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00537</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00535</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00552</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00541</span></span>
<span class="line"><span>Validation: Loss 0.00522 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00501 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56316</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51362</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.46587</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44222</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42381</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40441</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.37289</span></span>
<span class="line"><span>Validation: Loss 0.36375 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37596 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36591</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35794</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33095</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31272</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29893</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28205</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26940</span></span>
<span class="line"><span>Validation: Loss 0.25448 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26255 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26247</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24280</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22428</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22090</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20664</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19857</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18688</span></span>
<span class="line"><span>Validation: Loss 0.17701 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18136 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17990</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16881</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15792</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15347</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14881</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13767</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13352</span></span>
<span class="line"><span>Validation: Loss 0.12562 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12842 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12649</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12160</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11529</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10803</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10416</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10049</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09833</span></span>
<span class="line"><span>Validation: Loss 0.09088 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09327 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09061</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08727</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08393</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08035</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07851</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07324</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.06634</span></span>
<span class="line"><span>Validation: Loss 0.06677 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06906 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06702</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06306</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06158</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05855</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05752</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05597</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05555</span></span>
<span class="line"><span>Validation: Loss 0.04964 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05192 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05171</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04802</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04646</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04366</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04213</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04038</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04080</span></span>
<span class="line"><span>Validation: Loss 0.03730 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03938 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03813</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03492</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03649</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03446</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03215</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03066</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02812</span></span>
<span class="line"><span>Validation: Loss 0.02848 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03037 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02945</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02742</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02668</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02618</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02480</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02522</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02184</span></span>
<span class="line"><span>Validation: Loss 0.02240 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02410 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02170</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02274</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02153</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02086</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02064</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01961</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01914</span></span>
<span class="line"><span>Validation: Loss 0.01826 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01976 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01989</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01881</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01743</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01728</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01607</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01578</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01522</span></span>
<span class="line"><span>Validation: Loss 0.01537 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01665 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01595</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01666</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01449</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01410</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01408</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01405</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01344</span></span>
<span class="line"><span>Validation: Loss 0.01329 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01441 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01378</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01252</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01302</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01416</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01236</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01199</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01187</span></span>
<span class="line"><span>Validation: Loss 0.01173 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01274 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01182</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01199</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01163</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01164</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01108</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01065</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01178</span></span>
<span class="line"><span>Validation: Loss 0.01051 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01142 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01028</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01061</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01089</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01032</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01043</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00973</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00928</span></span>
<span class="line"><span>Validation: Loss 0.00952 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01035 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01009</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00878</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00914</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00925</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00957</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00901</span></span>
<span class="line"><span>Validation: Loss 0.00870 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00946 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00903</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00911</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00830</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00894</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00794</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00844</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00805</span></span>
<span class="line"><span>Validation: Loss 0.00800 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00870 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00835</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00813</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00747</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00764</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00796</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00796</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00811</span></span>
<span class="line"><span>Validation: Loss 0.00739 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00805 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00702</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00693</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00786</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00733</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00757</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00738</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00711</span></span>
<span class="line"><span>Validation: Loss 0.00687 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00748 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00703</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00702</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00724</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00727</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00655</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00613</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00582</span></span>
<span class="line"><span>Validation: Loss 0.00640 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00698 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00627</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00615</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00663</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00662</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00634</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00615</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00671</span></span>
<span class="line"><span>Validation: Loss 0.00599 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00654 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00605</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00592</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00615</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00598</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00615</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00559</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00600</span></span>
<span class="line"><span>Validation: Loss 0.00562 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00614 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00575</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00558</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00621</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00531</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00548</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00552</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00496</span></span>
<span class="line"><span>Validation: Loss 0.00529 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00578 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00526</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00550</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00500</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00541</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00527</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00532</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00514</span></span>
<span class="line"><span>Validation: Loss 0.00499 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00545 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.484 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,c,k,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
