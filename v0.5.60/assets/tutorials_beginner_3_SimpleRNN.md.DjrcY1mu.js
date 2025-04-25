import{_ as s,c as a,o as n,a4 as i}from"./chunks/framework.f2EDVGEo.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/v0.5.60/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/v0.5.60/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/v0.5.60/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/v0.5.60/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56355</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50409</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47888</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45083</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42343</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40456</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.38612</span></span>
<span class="line"><span>Validation: Loss 0.36356 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37005 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36180</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35023</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33715</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31982</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30149</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28226</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28104</span></span>
<span class="line"><span>Validation: Loss 0.25440 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25867 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26088</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24458</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23620</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22123</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20626</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19577</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18755</span></span>
<span class="line"><span>Validation: Loss 0.17801 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18026 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17920</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17105</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16305</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15478</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14668</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14200</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13249</span></span>
<span class="line"><span>Validation: Loss 0.12704 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12873 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12812</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12163</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11613</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11162</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10648</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10322</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09791</span></span>
<span class="line"><span>Validation: Loss 0.09211 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09376 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09388</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09062</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08517</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08153</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07829</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07281</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07301</span></span>
<span class="line"><span>Validation: Loss 0.06780 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06935 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06827</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06637</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06191</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06204</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05937</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05533</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05172</span></span>
<span class="line"><span>Validation: Loss 0.05048 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05189 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05405</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04732</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04734</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04677</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04291</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04163</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03814</span></span>
<span class="line"><span>Validation: Loss 0.03793 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03919 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03991</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03705</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03550</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03373</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03318</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03190</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03262</span></span>
<span class="line"><span>Validation: Loss 0.02893 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03005 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02775</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02919</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02919</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02641</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02592</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02507</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02401</span></span>
<span class="line"><span>Validation: Loss 0.02271 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02368 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02313</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02292</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02160</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02212</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02151</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01938</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01799</span></span>
<span class="line"><span>Validation: Loss 0.01845 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01928 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01888</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01847</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01803</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01736</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01838</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01629</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01587</span></span>
<span class="line"><span>Validation: Loss 0.01550 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01622 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01608</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01489</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01495</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01528</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01504</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01516</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01285</span></span>
<span class="line"><span>Validation: Loss 0.01339 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01402 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01401</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01306</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01331</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01331</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01327</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01250</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01204</span></span>
<span class="line"><span>Validation: Loss 0.01180 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01236 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01242</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01127</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01177</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01127</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01166</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01166</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01236</span></span>
<span class="line"><span>Validation: Loss 0.01056 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01107 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01043</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01021</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01078</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01052</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01049</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01075</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01023</span></span>
<span class="line"><span>Validation: Loss 0.00956 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01002 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00935</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01002</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01009</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00940</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00962</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00902</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00888</span></span>
<span class="line"><span>Validation: Loss 0.00872 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00914 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00941</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00894</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00821</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00879</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00890</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00827</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00858</span></span>
<span class="line"><span>Validation: Loss 0.00801 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00840 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00831</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00818</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00791</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00809</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00770</span></span>
<span class="line"><span>Validation: Loss 0.00740 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00776 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00773</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00790</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00724</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00726</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00746</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00719</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00718</span></span>
<span class="line"><span>Validation: Loss 0.00686 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00720 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00688</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00676</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00705</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00719</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00690</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00689</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00649</span></span>
<span class="line"><span>Validation: Loss 0.00640 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00672 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00698</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00608</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00627</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00623</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00626</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00711</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00587</span></span>
<span class="line"><span>Validation: Loss 0.00598 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00628 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00634</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00588</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00597</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00586</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00620</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00604</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00624</span></span>
<span class="line"><span>Validation: Loss 0.00561 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00590 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00563</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00579</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00565</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00560</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00566</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00580</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00569</span></span>
<span class="line"><span>Validation: Loss 0.00528 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00555 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00555</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00542</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00512</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00540</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00548</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00529</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00502</span></span>
<span class="line"><span>Validation: Loss 0.00498 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00523 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56278</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50906</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47558</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.43886</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42507</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40520</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.37820</span></span>
<span class="line"><span>Validation: Loss 0.37657 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.37322 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35897</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35401</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33382</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30988</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30473</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28282</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26542</span></span>
<span class="line"><span>Validation: Loss 0.26327 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.26085 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25836</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24372</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23031</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22172</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21010</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19294</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18100</span></span>
<span class="line"><span>Validation: Loss 0.18219 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.18084 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17912</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17051</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16191</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15302</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14660</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13853</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.12825</span></span>
<span class="line"><span>Validation: Loss 0.12957 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12861 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12761</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11913</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11583</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11137</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10415</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10068</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09525</span></span>
<span class="line"><span>Validation: Loss 0.09451 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09361 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09383</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08609</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08407</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08004</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07824</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07189</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07193</span></span>
<span class="line"><span>Validation: Loss 0.07012 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06928 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06907</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06498</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06020</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05886</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05868</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05423</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05098</span></span>
<span class="line"><span>Validation: Loss 0.05260 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05184 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05065</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04961</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04622</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04531</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04274</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03862</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03899</span></span>
<span class="line"><span>Validation: Loss 0.03982 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03915 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03867</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03633</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03401</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03298</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03246</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03170</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02889</span></span>
<span class="line"><span>Validation: Loss 0.03067 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03007 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02896</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02799</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02618</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02598</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02492</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02452</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02361</span></span>
<span class="line"><span>Validation: Loss 0.02430 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02378 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02283</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02288</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02030</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02112</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01976</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01951</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01869</span></span>
<span class="line"><span>Validation: Loss 0.01986 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01941 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01875</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01747</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01752</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01742</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01564</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01662</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01830</span></span>
<span class="line"><span>Validation: Loss 0.01672 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01633 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01529</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01587</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01464</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01504</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01380</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01401</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01239</span></span>
<span class="line"><span>Validation: Loss 0.01442 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01409 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01304</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01359</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01253</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01324</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01212</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01274</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01022</span></span>
<span class="line"><span>Validation: Loss 0.01272 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01243 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01107</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01214</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01167</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01152</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01099</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01065</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01098</span></span>
<span class="line"><span>Validation: Loss 0.01140 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01113 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00979</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01035</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01063</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01004</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01017</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01006</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01032</span></span>
<span class="line"><span>Validation: Loss 0.01033 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01008 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00957</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00950</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00918</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00920</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00875</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00938</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00872</span></span>
<span class="line"><span>Validation: Loss 0.00942 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00919 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00897</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00857</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00864</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00883</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00784</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00808</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00772</span></span>
<span class="line"><span>Validation: Loss 0.00866 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00845 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00799</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00765</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00790</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00804</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00755</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00753</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00802</span></span>
<span class="line"><span>Validation: Loss 0.00801 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00781 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00751</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00743</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00698</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00662</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00743</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00723</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00729</span></span>
<span class="line"><span>Validation: Loss 0.00743 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00725 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00700</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00701</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00640</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00645</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00660</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00687</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00620</span></span>
<span class="line"><span>Validation: Loss 0.00693 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00675 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00607</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00675</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00642</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00640</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00592</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00602</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00598</span></span>
<span class="line"><span>Validation: Loss 0.00648 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00632 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00613</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00597</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00586</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00573</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00564</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00593</span></span>
<span class="line"><span>Validation: Loss 0.00608 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00593 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00555</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00583</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00551</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00534</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00530</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00559</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00497</span></span>
<span class="line"><span>Validation: Loss 0.00573 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00558 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00544</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00520</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00527</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00494</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00490</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00529</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00527</span></span>
<span class="line"><span>Validation: Loss 0.00541 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00527 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>- CUDA_Driver_jll: 0.9.1+1</span></span>
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
