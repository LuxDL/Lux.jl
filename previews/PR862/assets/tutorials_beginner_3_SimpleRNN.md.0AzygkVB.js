import{_ as s,c as a,o as n,a4 as i}from"./chunks/framework.D-jK4gJk.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR862/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR862/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR862/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR862/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56460</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51762</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47127</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44117</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.43370</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40625</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40276</span></span>
<span class="line"><span>Validation: Loss 0.35809 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.35774 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37427</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35636</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33479</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31839</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30927</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28517</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.26548</span></span>
<span class="line"><span>Validation: Loss 0.25340 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25312 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26444</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24926</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23199</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22161</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21070</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20084</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18894</span></span>
<span class="line"><span>Validation: Loss 0.17880 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17865 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18126</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17351</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16617</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15597</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14892</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14312</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13326</span></span>
<span class="line"><span>Validation: Loss 0.12791 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12783 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.13012</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12391</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11852</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11430</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10679</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10295</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09946</span></span>
<span class="line"><span>Validation: Loss 0.09236 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09228 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09431</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09027</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08627</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08311</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07906</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07704</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07423</span></span>
<span class="line"><span>Validation: Loss 0.06731 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06723 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07056</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06732</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06400</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06199</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05976</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05559</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05198</span></span>
<span class="line"><span>Validation: Loss 0.04961 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04954 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05100</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04883</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04717</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04704</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04621</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04385</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03904</span></span>
<span class="line"><span>Validation: Loss 0.03679 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03671 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03745</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03822</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03741</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03595</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03318</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03275</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03088</span></span>
<span class="line"><span>Validation: Loss 0.02778 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02771 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02985</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03042</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02761</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02627</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02597</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02532</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02825</span></span>
<span class="line"><span>Validation: Loss 0.02171 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02165 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02363</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02342</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02198</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02045</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02208</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02128</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02003</span></span>
<span class="line"><span>Validation: Loss 0.01762 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01757 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.02037</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01809</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01809</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01847</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01771</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01739</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01526</span></span>
<span class="line"><span>Validation: Loss 0.01481 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01475 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01663</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01551</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01531</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01579</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01475</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01526</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01496</span></span>
<span class="line"><span>Validation: Loss 0.01279 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01274 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01460</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01411</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01359</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01397</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01266</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01262</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01256</span></span>
<span class="line"><span>Validation: Loss 0.01127 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01123 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01303</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01246</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01106</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01193</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01224</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01135</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01245</span></span>
<span class="line"><span>Validation: Loss 0.01008 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01005 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01109</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01115</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01092</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01081</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01093</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01024</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01008</span></span>
<span class="line"><span>Validation: Loss 0.00912 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00908 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01024</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00989</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00972</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00978</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00954</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00971</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01029</span></span>
<span class="line"><span>Validation: Loss 0.00832 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00828 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00956</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00859</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00939</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00875</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00873</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00906</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00870</span></span>
<span class="line"><span>Validation: Loss 0.00763 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00760 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00843</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00906</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00787</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00837</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00811</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00797</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00807</span></span>
<span class="line"><span>Validation: Loss 0.00704 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00701 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00833</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00752</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00778</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00705</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00761</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00755</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00852</span></span>
<span class="line"><span>Validation: Loss 0.00653 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00650 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00744</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00752</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00708</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00719</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00709</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00654</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00695</span></span>
<span class="line"><span>Validation: Loss 0.00609 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00606 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00674</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00641</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00703</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00694</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00663</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00635</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00612</span></span>
<span class="line"><span>Validation: Loss 0.00568 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00566 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00615</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00641</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00637</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00599</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00644</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00604</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00630</span></span>
<span class="line"><span>Validation: Loss 0.00533 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00531 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00603</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00590</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00585</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00557</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00619</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00573</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00542</span></span>
<span class="line"><span>Validation: Loss 0.00501 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00499 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00569</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00535</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00524</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00575</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00579</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00542</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00508</span></span>
<span class="line"><span>Validation: Loss 0.00473 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00471 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56179</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51324</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47383</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44703</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42463</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40459</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39724</span></span>
<span class="line"><span>Validation: Loss 0.36855 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.35955 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.37125</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34594</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33314</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31989</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29866</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28374</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28198</span></span>
<span class="line"><span>Validation: Loss 0.25748 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25238 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25863</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24234</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23126</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22349</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20897</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19732</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18975</span></span>
<span class="line"><span>Validation: Loss 0.17949 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17728 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17958</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16893</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16364</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15486</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14839</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14082</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13422</span></span>
<span class="line"><span>Validation: Loss 0.12792 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12648 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12852</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12232</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11711</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11096</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10590</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10129</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09897</span></span>
<span class="line"><span>Validation: Loss 0.09288 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09139 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09152</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08917</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08560</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08167</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07959</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07425</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.06871</span></span>
<span class="line"><span>Validation: Loss 0.06847 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06680 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06847</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06477</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06319</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06078</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05826</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05502</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05506</span></span>
<span class="line"><span>Validation: Loss 0.05109 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04933 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05061</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04799</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04690</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04658</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04421</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04112</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04048</span></span>
<span class="line"><span>Validation: Loss 0.03848 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03680 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03930</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03630</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03461</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03449</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03215</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03283</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03094</span></span>
<span class="line"><span>Validation: Loss 0.02945 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02790 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03038</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02767</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02821</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02565</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02611</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02397</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02384</span></span>
<span class="line"><span>Validation: Loss 0.02321 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02184 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02417</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02227</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02074</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02157</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01982</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02064</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01881</span></span>
<span class="line"><span>Validation: Loss 0.01892 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01773 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01899</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01798</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01808</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01742</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01686</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01693</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01663</span></span>
<span class="line"><span>Validation: Loss 0.01593 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01488 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01564</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01589</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01525</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01459</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01463</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01431</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01423</span></span>
<span class="line"><span>Validation: Loss 0.01377 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01285 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01421</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01323</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01318</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01291</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01261</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01251</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01251</span></span>
<span class="line"><span>Validation: Loss 0.01214 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01132 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01181</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01156</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01206</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01099</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01136</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01198</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01096</span></span>
<span class="line"><span>Validation: Loss 0.01087 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01013 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01041</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01051</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01085</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01083</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00987</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01017</span></span>
<span class="line"><span>Validation: Loss 0.00984 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00916 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00938</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01019</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00986</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00858</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00925</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00958</span></span>
<span class="line"><span>Validation: Loss 0.00899 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00836 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00922</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00868</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00816</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00859</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00891</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00823</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00967</span></span>
<span class="line"><span>Validation: Loss 0.00826 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00768 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00816</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00816</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00829</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00756</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00809</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00768</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00786</span></span>
<span class="line"><span>Validation: Loss 0.00763 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00709 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00823</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00690</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00734</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00759</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00762</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00691</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00648</span></span>
<span class="line"><span>Validation: Loss 0.00708 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00657 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00717</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00673</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00684</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00696</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00717</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00657</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00600</span></span>
<span class="line"><span>Validation: Loss 0.00660 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00613 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00674</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00652</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00643</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00633</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00624</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00625</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00635</span></span>
<span class="line"><span>Validation: Loss 0.00618 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00573 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00613</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00606</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00592</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00584</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00605</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00620</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00544</span></span>
<span class="line"><span>Validation: Loss 0.00580 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00538 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00563</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00544</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00591</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00549</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00580</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00571</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00528</span></span>
<span class="line"><span>Validation: Loss 0.00546 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00506 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00554</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00579</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00532</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00513</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00530</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00499</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00480</span></span>
<span class="line"><span>Validation: Loss 0.00515 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00477 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
<span class="line"><span>- Julia: 1.10.5</span></span>
<span class="line"><span>- LLVM: 15.0.7</span></span>
<span class="line"><span></span></span>
<span class="line"><span>Environment:</span></span>
<span class="line"><span>- JULIA_CUDA_HARD_MEMORY_LIMIT: 100%</span></span>
<span class="line"><span></span></span>
<span class="line"><span>1 device:</span></span>
<span class="line"><span>  0: NVIDIA A100-PCIE-40GB MIG 1g.5gb (sm_80, 4.453 GiB / 4.750 GiB available)</span></span></code></pre></div><hr><p><em>This page was generated using <a href="https://github.com/fredrikekre/Literate.jl" target="_blank" rel="noreferrer">Literate.jl</a>.</em></p>`,45),h=[l];function e(t,c,k,o,E,r){return n(),a("div",null,h)}const g=s(p,[["render",e]]);export{y as __pageData,g as default};
