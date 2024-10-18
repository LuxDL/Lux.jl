import{_ as s,c as a,o as n,a4 as i}from"./chunks/framework.GhNgc9YX.js";const y=JSON.parse('{"title":"Training a Simple LSTM","description":"","frontmatter":{},"headers":[],"relativePath":"tutorials/beginner/3_SimpleRNN.md","filePath":"tutorials/beginner/3_SimpleRNN.md","lastUpdated":null}'),p={name:"tutorials/beginner/3_SimpleRNN.md"},l=i(`<h1 id="Training-a-Simple-LSTM" tabindex="-1">Training a Simple LSTM <a class="header-anchor" href="#Training-a-Simple-LSTM" aria-label="Permalink to &quot;Training a Simple LSTM {#Training-a-Simple-LSTM}&quot;">​</a></h1><p>In this tutorial we will go over using a recurrent neural network to classify clockwise and anticlockwise spirals. By the end of this tutorial you will be able to:</p><ol><li><p>Create custom Lux models.</p></li><li><p>Become familiar with the Lux recurrent neural network API.</p></li><li><p>Training using Optimisers.jl and Zygote.jl.</p></li></ol><h2 id="Package-Imports" tabindex="-1">Package Imports <a class="header-anchor" href="#Package-Imports" aria-label="Permalink to &quot;Package Imports {#Package-Imports}&quot;">​</a></h2><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ADTypes, Lux, AMDGPU, LuxCUDA, JLD2, MLUtils, Optimisers, Zygote, Printf, Random,</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>get_dataloaders (generic function with 1 method)</span></span></code></pre></div><h2 id="Creating-a-Classifier" tabindex="-1">Creating a Classifier <a class="header-anchor" href="#Creating-a-Classifier" aria-label="Permalink to &quot;Creating a Classifier {#Creating-a-Classifier}&quot;">​</a></h2><p>We will be extending the <code>Lux.AbstractExplicitContainerLayer</code> type for our custom model since it will contain a lstm block and a classifier head.</p><p>We pass the fieldnames <code>lstm_cell</code> and <code>classifier</code> to the type to ensure that the parameters and states are automatically populated and we don&#39;t have to define <code>Lux.initialparameters</code> and <code>Lux.initialstates</code>.</p><p>To understand more about container layers, please look at <a href="/previews/PR834/manual/interface#Container-Layer">Container Layer</a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">struct</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SpiralClassifier{L, C} </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">:</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">       Lux</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">.</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AbstractExplicitContainerLayer{(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:lstm_cell</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:classifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)}</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    lstm_cell</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">L</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    classifier</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">C</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><p>We won&#39;t define the model from scratch but rather use the <a href="/previews/PR834/api/Lux/layers#Lux.LSTMCell"><code>Lux.LSTMCell</code></a> and <a href="/previews/PR834/api/Lux/layers#Lux.Dense"><code>Lux.Dense</code></a>.</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifier</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><h2 id="Using-the-@compact-API" tabindex="-1">Using the <code>@compact</code> API <a class="header-anchor" href="#Using-the-@compact-API" aria-label="Permalink to &quot;Using the \`@compact\` API {#Using-the-@compact-API}&quot;">​</a></h2><p>We can also define the model using the <a href="/previews/PR834/api/Lux/utilities#Lux.@compact"><code>Lux.@compact</code></a> API, which is a more concise way of defining models. This macro automatically handles the boilerplate code for you and as such we recommend this way of defining custom layers</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">function</span><span style="--shiki-light:#6F42C1;--shiki-dark:#B392F0;"> SpiralClassifierCompact</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(in_dims, hidden_dims, out_dims)</span></span>
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
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained, st_trained </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifier)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56277</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.50741</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47774</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.45273</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42471</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39978</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.39699</span></span>
<span class="line"><span>Validation: Loss 0.35864 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.36432 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36721</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.34876</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33533</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31934</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.29964</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28729</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.27043</span></span>
<span class="line"><span>Validation: Loss 0.25223 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25596 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.25837</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24373</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23169</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22218</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20934</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19873</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.18514</span></span>
<span class="line"><span>Validation: Loss 0.17694 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17901 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17853</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17213</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16377</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15372</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14709</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14086</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13209</span></span>
<span class="line"><span>Validation: Loss 0.12610 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12756 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12764</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12143</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11546</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11151</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10733</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10175</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09876</span></span>
<span class="line"><span>Validation: Loss 0.09090 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09233 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09151</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08845</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08434</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08087</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07981</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07513</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07397</span></span>
<span class="line"><span>Validation: Loss 0.06648 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06782 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.07056</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06512</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06218</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06114</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05706</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05502</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05289</span></span>
<span class="line"><span>Validation: Loss 0.04924 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.05039 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05081</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04918</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04873</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04526</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04303</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04058</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.03985</span></span>
<span class="line"><span>Validation: Loss 0.03671 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03773 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03723</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03781</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03483</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03494</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03359</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03123</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03131</span></span>
<span class="line"><span>Validation: Loss 0.02777 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02869 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02907</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02793</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02817</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02664</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02532</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02460</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02478</span></span>
<span class="line"><span>Validation: Loss 0.02171 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02251 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02243</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02301</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02109</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02283</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01946</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02049</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.01868</span></span>
<span class="line"><span>Validation: Loss 0.01762 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01831 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01849</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01843</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01866</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01717</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01690</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01699</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01570</span></span>
<span class="line"><span>Validation: Loss 0.01478 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01539 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01587</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01567</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01486</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01534</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01499</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01399</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01346</span></span>
<span class="line"><span>Validation: Loss 0.01277 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01330 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01418</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01347</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01377</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01246</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01248</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01277</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01152</span></span>
<span class="line"><span>Validation: Loss 0.01125 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01174 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01143</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01124</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01208</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01198</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01165</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01165</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01089</span></span>
<span class="line"><span>Validation: Loss 0.01007 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01051 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01077</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01073</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01043</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00965</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01105</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01027</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01042</span></span>
<span class="line"><span>Validation: Loss 0.00912 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00952 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00983</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00975</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00941</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00945</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01040</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00861</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00831</span></span>
<span class="line"><span>Validation: Loss 0.00831 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00869 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00909</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00920</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00943</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00837</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00832</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00816</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00780</span></span>
<span class="line"><span>Validation: Loss 0.00763 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00798 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00849</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00846</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00763</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00803</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00775</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00797</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00743</span></span>
<span class="line"><span>Validation: Loss 0.00705 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00737 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00777</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00742</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00767</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00716</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00686</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00783</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00706</span></span>
<span class="line"><span>Validation: Loss 0.00654 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00685 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00721</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00718</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00671</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00665</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00694</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00687</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00665</span></span>
<span class="line"><span>Validation: Loss 0.00609 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00638 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00649</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00669</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00626</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00648</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00666</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00618</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00635</span></span>
<span class="line"><span>Validation: Loss 0.00570 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00597 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00631</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00597</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00645</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00594</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00582</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00571</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00645</span></span>
<span class="line"><span>Validation: Loss 0.00534 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00560 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00592</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00574</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00563</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00576</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00558</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00539</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00595</span></span>
<span class="line"><span>Validation: Loss 0.00503 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00527 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00531</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00560</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00570</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00520</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00501</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00538</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00494</span></span>
<span class="line"><span>Validation: Loss 0.00474 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00497 Accuracy 1.00000</span></span></code></pre></div><p>We can also train the compact model with the exact same code!</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ps_trained2, st_trained2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> main</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(SpiralClassifierCompact)</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Epoch [  1]: Loss 0.56246</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.51592</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.47317</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.44816</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.42839</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.40135</span></span>
<span class="line"><span>Epoch [  1]: Loss 0.35970</span></span>
<span class="line"><span>Validation: Loss 0.36845 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.35922 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.36465</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.35212</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.33808</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.31808</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.30014</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28424</span></span>
<span class="line"><span>Epoch [  2]: Loss 0.28557</span></span>
<span class="line"><span>Validation: Loss 0.25887 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.25233 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.26018</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.24353</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.23236</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.22053</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.21015</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.20058</span></span>
<span class="line"><span>Epoch [  3]: Loss 0.19517</span></span>
<span class="line"><span>Validation: Loss 0.18020 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.17698 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.18113</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.17014</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.16329</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.15470</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14947</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.14088</span></span>
<span class="line"><span>Epoch [  4]: Loss 0.13429</span></span>
<span class="line"><span>Validation: Loss 0.12824 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.12644 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12736</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.12226</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11616</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.11402</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10697</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.10212</span></span>
<span class="line"><span>Epoch [  5]: Loss 0.09560</span></span>
<span class="line"><span>Validation: Loss 0.09318 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.09155 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.09324</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08849</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08466</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.08108</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07918</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07600</span></span>
<span class="line"><span>Epoch [  6]: Loss 0.07362</span></span>
<span class="line"><span>Validation: Loss 0.06879 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.06705 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06857</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06477</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06114</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.06211</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05778</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05748</span></span>
<span class="line"><span>Epoch [  7]: Loss 0.05805</span></span>
<span class="line"><span>Validation: Loss 0.05140 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.04969 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05252</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.05135</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04663</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04505</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04316</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04083</span></span>
<span class="line"><span>Epoch [  8]: Loss 0.04094</span></span>
<span class="line"><span>Validation: Loss 0.03878 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.03721 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03969</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03803</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03589</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03470</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03230</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.03140</span></span>
<span class="line"><span>Epoch [  9]: Loss 0.02999</span></span>
<span class="line"><span>Validation: Loss 0.02971 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02825 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.03070</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02829</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02769</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02587</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02559</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02546</span></span>
<span class="line"><span>Epoch [ 10]: Loss 0.02404</span></span>
<span class="line"><span>Validation: Loss 0.02342 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.02208 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02176</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02369</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02140</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02222</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02045</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02056</span></span>
<span class="line"><span>Epoch [ 11]: Loss 0.02063</span></span>
<span class="line"><span>Validation: Loss 0.01910 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01792 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01923</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01829</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01841</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01758</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01681</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01758</span></span>
<span class="line"><span>Epoch [ 12]: Loss 0.01500</span></span>
<span class="line"><span>Validation: Loss 0.01608 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01503 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01556</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01686</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01551</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01551</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01406</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01398</span></span>
<span class="line"><span>Epoch [ 13]: Loss 0.01406</span></span>
<span class="line"><span>Validation: Loss 0.01390 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01298 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01431</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01399</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01295</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01316</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01260</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01218</span></span>
<span class="line"><span>Epoch [ 14]: Loss 0.01453</span></span>
<span class="line"><span>Validation: Loss 0.01228 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01144 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01221</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01245</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01150</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01244</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01030</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01175</span></span>
<span class="line"><span>Epoch [ 15]: Loss 0.01138</span></span>
<span class="line"><span>Validation: Loss 0.01099 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.01024 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01117</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01091</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01108</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01004</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01031</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.01025</span></span>
<span class="line"><span>Epoch [ 16]: Loss 0.00944</span></span>
<span class="line"><span>Validation: Loss 0.00996 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00926 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00987</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00955</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.01058</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00966</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00846</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00959</span></span>
<span class="line"><span>Epoch [ 17]: Loss 0.00945</span></span>
<span class="line"><span>Validation: Loss 0.00909 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00845 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00865</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00921</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00916</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00889</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00848</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00838</span></span>
<span class="line"><span>Epoch [ 18]: Loss 0.00894</span></span>
<span class="line"><span>Validation: Loss 0.00836 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00776 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00850</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00780</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00856</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00796</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00830</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00749</span></span>
<span class="line"><span>Epoch [ 19]: Loss 0.00827</span></span>
<span class="line"><span>Validation: Loss 0.00773 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00717 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00750</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00772</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00734</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00779</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00742</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00724</span></span>
<span class="line"><span>Epoch [ 20]: Loss 0.00762</span></span>
<span class="line"><span>Validation: Loss 0.00718 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00665 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00689</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00701</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00709</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00676</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00701</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00725</span></span>
<span class="line"><span>Epoch [ 21]: Loss 0.00651</span></span>
<span class="line"><span>Validation: Loss 0.00669 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00620 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00695</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00642</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00642</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00631</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00670</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00628</span></span>
<span class="line"><span>Epoch [ 22]: Loss 0.00671</span></span>
<span class="line"><span>Validation: Loss 0.00626 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00580 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00619</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00640</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00610</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00610</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00591</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00608</span></span>
<span class="line"><span>Epoch [ 23]: Loss 0.00556</span></span>
<span class="line"><span>Validation: Loss 0.00588 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00544 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00571</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00570</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00555</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00591</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00589</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00580</span></span>
<span class="line"><span>Epoch [ 24]: Loss 0.00520</span></span>
<span class="line"><span>Validation: Loss 0.00553 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00512 Accuracy 1.00000</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00524</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00580</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00516</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00574</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00524</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00528</span></span>
<span class="line"><span>Epoch [ 25]: Loss 0.00542</span></span>
<span class="line"><span>Validation: Loss 0.00522 Accuracy 1.00000</span></span>
<span class="line"><span>Validation: Loss 0.00483 Accuracy 1.00000</span></span></code></pre></div><h2 id="Saving-the-Model" tabindex="-1">Saving the Model <a class="header-anchor" href="#Saving-the-Model" aria-label="Permalink to &quot;Saving the Model {#Saving-the-Model}&quot;">​</a></h2><p>We can save the model using JLD2 (and any other serialization library of your choice) Note that we transfer the model to CPU before saving. Additionally, we recommend that you don&#39;t save the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@save</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {compress </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> true</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">} ps_trained st_trained</span></span></code></pre></div><p>Let&#39;s try loading the model</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@load</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> &quot;trained_model.jld2&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ps_trained st_trained</span></span></code></pre></div><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>2-element Vector{Symbol}:</span></span>
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
