import{_ as s,c as a,o as n,a3 as i}from"./chunks/framework.CgGBq5OQ.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v0.5.55/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v0.5.55/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v0.5.55/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v0.5.55/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>SpiralClassifierCompact (generic function with 1 method)</span></span></code></pre></div><h2 id="Defining-Accuracy,-Loss-and-Optimiser" tabindex="-1">Defining Accuracy, Loss and Optimiser <a class="header-anchor" href="#Defining-Accuracy,-Loss-and-Optimiser" aria-label="Permalink to &quot;Defining Accuracy, Loss and Optimiser {#Defining-Accuracy,-Loss-and-Optimiser}&quot;">​</a></h2><p>Now let&#39;s define the binarycrossentropy loss. Typically it is recommended to use <code>logitbinarycrossentropy</code> since it is more numerically stable, but for the sake of simplicity we will use <code>binarycrossentropy</code>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> xlogy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    result </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> ifelse</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">iszero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">zero</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(result), result)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> binarycrossentropy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y_true)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_pred </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.+</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> eps</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">eltype</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> mean</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@.</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">xlogy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_true, y_pred) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> xlogy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_true, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> -</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_pred))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, ps, st, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_pred, st </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> model</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x, ps, st)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> binarycrossentropy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_pred, y), st, (; y_pred</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y_pred)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    train_state </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TrainState</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        rng, model, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Adam</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.01f0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">); transform_variables</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">dev)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">25</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Train the model</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> train_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            (_, loss, _, train_state) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Experimental</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">single_train_step!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">                AutoZygote</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), compute_loss, (x, y), train_state)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Epoch [%3d]: Loss %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> epoch loss</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">        # Validate the model</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">        st_ </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">testmode</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (x, y) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">in</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> val_loader</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> dev</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            loss, st_, ret </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> compute_loss</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(model, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, st_, (x, y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">            acc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> accuracy</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(ret</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y_pred, y)</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">            @printf</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;Validation: Loss %4.5f Accuracy %4.5f</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">\\n</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> loss acc</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">        end</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">    return</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">parameters, train_state</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">states) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">|&gt;</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> cpu_device</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56195</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51227</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.48014</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44802</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42916</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40174</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.36822</span></span>
<span class="line"><span>Validation: Loss 0.36514 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36982 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37403</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34872</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33520</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31769</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29522</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28634</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28288</span></span>
<span class="line"><span>Validation: Loss 0.25600 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25940 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25895</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24491</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23783</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22123</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20754</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19956</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18468</span></span>
<span class="line"><span>Validation: Loss 0.17877 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18065 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18289</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17219</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16199</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15496</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14764</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14167</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13067</span></span>
<span class="line"><span>Validation: Loss 0.12742 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12876 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12759</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12416</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11736</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11165</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10764</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10039</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09892</span></span>
<span class="line"><span>Validation: Loss 0.09246 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09373 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09272</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09057</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08440</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08252</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07830</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07558</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07135</span></span>
<span class="line"><span>Validation: Loss 0.06813 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06934 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07041</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06683</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06391</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06077</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05725</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05545</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05138</span></span>
<span class="line"><span>Validation: Loss 0.05078 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05189 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05004</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05048</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04867</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04407</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04422</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04261</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04115</span></span>
<span class="line"><span>Validation: Loss 0.03818 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03919 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03731</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03662</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03793</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03474</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03338</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03263</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02944</span></span>
<span class="line"><span>Validation: Loss 0.02914 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03004 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02907</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02834</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02673</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02810</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02598</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02578</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02363</span></span>
<span class="line"><span>Validation: Loss 0.02291 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02369 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02384</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02354</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02100</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02134</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02144</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01962</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01913</span></span>
<span class="line"><span>Validation: Loss 0.01864 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01931 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01885</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01887</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01784</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01687</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01814</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01736</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01508</span></span>
<span class="line"><span>Validation: Loss 0.01567 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01625 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01580</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01529</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01513</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01502</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01504</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01530</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01336</span></span>
<span class="line"><span>Validation: Loss 0.01354 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01405 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01416</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01348</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01410</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01276</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01298</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01230</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01205</span></span>
<span class="line"><span>Validation: Loss 0.01194 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01239 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01258</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01179</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01198</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01156</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01171</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01088</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01172</span></span>
<span class="line"><span>Validation: Loss 0.01068 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01109 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01064</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01127</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01030</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01056</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00971</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01063</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01149</span></span>
<span class="line"><span>Validation: Loss 0.00967 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01004 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00990</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01005</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00966</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00907</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00913</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00966</span></span>
<span class="line"><span>Validation: Loss 0.00882 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00916 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00880</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00944</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00861</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00841</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00868</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00792</span></span>
<span class="line"><span>Validation: Loss 0.00810 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00842 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00817</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00842</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00786</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00839</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00766</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00807</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00755</span></span>
<span class="line"><span>Validation: Loss 0.00748 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00778 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00796</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00759</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00705</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00750</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00697</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00821</span></span>
<span class="line"><span>Validation: Loss 0.00695 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00722 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00691</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00754</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00662</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00710</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00677</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00681</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00676</span></span>
<span class="line"><span>Validation: Loss 0.00647 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00673 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00695</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00672</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00636</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00615</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00656</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00623</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00624</span></span>
<span class="line"><span>Validation: Loss 0.00605 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00629 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00625</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00620</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00595</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00632</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00608</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00556</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00640</span></span>
<span class="line"><span>Validation: Loss 0.00568 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00591 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00594</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00555</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00585</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00549</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00613</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00540</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00518</span></span>
<span class="line"><span>Validation: Loss 0.00534 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00556 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00583</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00532</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00490</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00556</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00537</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00514</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00583</span></span>
<span class="line"><span>Validation: Loss 0.00504 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00524 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56059</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51384</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47283</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45049</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42934</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39947</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38569</span></span>
<span class="line"><span>Validation: Loss 0.36185 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36916 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.38349</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35008</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33138</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31695</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29258</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28621</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26666</span></span>
<span class="line"><span>Validation: Loss 0.25404 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25878 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25867</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24666</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23441</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22005</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20919</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19682</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18726</span></span>
<span class="line"><span>Validation: Loss 0.17754 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18029 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18019</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17244</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16427</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15370</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14584</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14073</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13221</span></span>
<span class="line"><span>Validation: Loss 0.12639 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12831 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12822</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12217</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11630</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11052</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10601</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10229</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09706</span></span>
<span class="line"><span>Validation: Loss 0.09140 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09324 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09220</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08827</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08558</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08232</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07684</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07568</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07025</span></span>
<span class="line"><span>Validation: Loss 0.06704 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06885 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06947</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06560</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06218</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06114</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05703</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05563</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05234</span></span>
<span class="line"><span>Validation: Loss 0.04974 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05142 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05092</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04712</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04893</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04503</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04153</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04313</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04335</span></span>
<span class="line"><span>Validation: Loss 0.03722 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03876 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03970</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03787</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03519</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03346</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03246</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03193</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02788</span></span>
<span class="line"><span>Validation: Loss 0.02831 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02966 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02915</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02857</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02805</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02603</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02639</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02373</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02439</span></span>
<span class="line"><span>Validation: Loss 0.02221 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02339 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02406</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02179</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02262</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01989</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02020</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02076</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01902</span></span>
<span class="line"><span>Validation: Loss 0.01806 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01907 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01989</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01870</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01809</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01698</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01673</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01633</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01588</span></span>
<span class="line"><span>Validation: Loss 0.01517 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01605 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01604</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01530</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01483</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01550</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01482</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01376</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01504</span></span>
<span class="line"><span>Validation: Loss 0.01311 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01388 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01286</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01304</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01471</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01269</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01272</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01288</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01193</span></span>
<span class="line"><span>Validation: Loss 0.01155 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01224 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01182</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01195</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01142</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01112</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01176</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01184</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01093</span></span>
<span class="line"><span>Validation: Loss 0.01034 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01096 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01127</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01130</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01035</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01041</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00996</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00970</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00961</span></span>
<span class="line"><span>Validation: Loss 0.00935 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00992 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01005</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00910</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00935</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00887</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00880</span></span>
<span class="line"><span>Validation: Loss 0.00853 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00906 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00899</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00875</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00920</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00818</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00935</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00790</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00783</span></span>
<span class="line"><span>Validation: Loss 0.00784 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00833 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00833</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00818</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00870</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00800</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00748</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00700</span></span>
<span class="line"><span>Validation: Loss 0.00724 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00770 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00813</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00720</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00763</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00710</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00680</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00755</span></span>
<span class="line"><span>Validation: Loss 0.00672 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00715 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00715</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00712</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00706</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00648</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00672</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00707</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00601</span></span>
<span class="line"><span>Validation: Loss 0.00627 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00667 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00689</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00635</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00654</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00646</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00601</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00636</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00655</span></span>
<span class="line"><span>Validation: Loss 0.00586 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00624 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00646</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00604</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00587</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00643</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00575</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00559</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00629</span></span>
<span class="line"><span>Validation: Loss 0.00550 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00586 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00599</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00551</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00557</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00589</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00527</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00564</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00613</span></span>
<span class="line"><span>Validation: Loss 0.00517 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00551 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00531</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00570</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00510</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00551</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00548</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00507</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00469</span></span>
<span class="line"><span>Validation: Loss 0.00488 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00520 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>  JULIA_AMDGPU_LOGGING_ENABLED = true</span></span>
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
<span class="line"><span>NVIDIA driver 555.42.2</span></span>
<span class="line"><span></span></span>
<span class="line"><span>CUDA libraries: </span></span>
<span class="line"><span>- CUBLAS: 12.5.2</span></span>
<span class="line"><span>- CURAND: 10.3.6</span></span>
<span class="line"><span>- CUFFT: 11.2.3</span></span>
<span class="line"><span>- CUSOLVER: 11.6.2</span></span>
<span class="line"><span>- CUSPARSE: 12.4.1</span></span>
<span class="line"><span>- CUPTI: 23.0.0</span></span>
<span class="line"><span>- NVML: 12.0.0+555.42.2</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Julia packages: </span></span>
<span class="line"><span>- CUDA: 5.4.2</span></span>
<span class="line"><span>- CUDA_Driver_jll: 0.9.0+0</span></span>
<span class="line"><span>- CUDA_Runtime_jll: 0.14.0+1</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Toolchain:</span></span>
<span class="line"><span>- Julia: 1.10.4</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.422 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,k,c,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
